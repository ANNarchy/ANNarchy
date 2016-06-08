"""

    Compiler.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import os, sys, imp
import subprocess
import shutil
import time
import numpy as np
import re

# ANNarchy core informations
import ANNarchy
import ANNarchy.core.Global as Global
from .Template.MakefileTemplate import *

from optparse import OptionParser
from optparse import OptionGroup

from distutils.sysconfig import get_python_inc, get_python_lib

# String containing the extra libs which can be added by extensions
# e.g. extra_libs = ['-lopencv_core', '-lopencv_video']
extra_libs = []

def _folder_management(annarchy_dir, profile_enabled, clean, net_id):
    """
    ANNarchy is provided as a python package. For compilation a local folder
    'annarchy' is created in the current working directory.

    *Parameter*:

    * annarchy_dir : subdirectory
    * *profile_enabled*: copy needed data for profile extension
    """

    # Verbose
    if Global.config['verbose']:
        Global._print("Create subdirectory.")

    if clean or profile_enabled:
        shutil.rmtree(annarchy_dir, True)

    # Create the subdirectory
    if not os.path.exists(annarchy_dir):
        os.mkdir(annarchy_dir)
        os.mkdir(annarchy_dir+'/build')
        os.mkdir(annarchy_dir+'/generate')

    # Subdirectory for building networks
    if not os.path.exists(annarchy_dir+'/build/net'+str(net_id)):
        os.mkdir(annarchy_dir+'/build/net'+str(net_id))

    # Create the generate subfolder
    if not os.path.exists(annarchy_dir+'/generate/net'+str(net_id)):
        os.mkdir(annarchy_dir+'/generate/net'+str(net_id))

    # Save current ANNarchy version
    with open(annarchy_dir+'/release', 'w') as f:
        f.write(ANNarchy.__release__)

    sys.path.append(annarchy_dir)

def setup_parser():
    # override the error behavior of OptionParser,
    # normally an unknwon arg would raise an exception
    class MyOptionParser(OptionParser):
        def error(self, msg):
            pass

    parser = MyOptionParser ("usage: python %prog [options]")

    group = OptionGroup(parser, "general")
    group.add_option("-c", "--clean", help="enforce complete recompile", action="store_true", default=False, dest="clean")
    group.add_option("-d", "--debug", help="ANNarchy is compiled with debug symbols and additional checks", action="store_true", default=False, dest="debug")
    group.add_option("-v", "--verbose", help="show all messages", action="store_true", default=None, dest="verbose")
    parser.add_option_group(group)

    group = OptionGroup(parser, "OpenMP")
    group.add_option("-j", help="number of threads should be used", type="int", action="store", default=None, dest="num_threads")
    parser.add_option_group(group)

    group = OptionGroup(parser, "others")
    group.add_option("--profile", help="enable profiling", action="store_true", default=None, dest="profile")
    parser.add_option_group(group)

    return parser

def compile(    directory='annarchy',
                clean=False,
                populations=None,
                projections=None,
                compiler="default",
                compiler_flags="-march=native -O2",
                cuda_config={'device': 0},
                silent=False,
                debug_build=False,
                profile_enabled = False,
                net_id=0 ):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will perform the simulation.

    *Parameters*:

    * **directory**: name of the subdirectory where the code will be generated and compiled. Must be a relative path.
    * **clean**: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * **populations**: list of populations which should be compiled. If set to None, all available populations will be used.
    * **projections**: list of projection which should be compiled. If set to None, all available projections will be used.
    * **compiler**: C++ compiler to use. Default: g++ on GNU/Linux, clang++ on OS X. Valid compilers are [g++, clang++].
    * **compiler_flags**: platform-specific flags to pass to the compiler. Default: "-march=native -O2". Warning: -O3 often generates slower code and can cause linking problems, so it is not recommended.
    * **cuda_config**: dictionary defining the CUDA configuration for each population and projection.
    * **silent**: defines if the "Compiling... OK" should be printed.

    The following arguments are for internal development use only:

    * **debug_build**: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    * **profile_enabled**: creates a profilable version of ANNarchy, which logs several computation timings (default: False).
    """
    parser = setup_parser()
    (options, args) = parser.parse_args()

    # if the parameters set on command-line they overwrite Global.config
    if options.num_threads != None:
        Global.config['num_threads'] = options.num_threads
    if options.verbose != None:
        Global.config['verbose'] = options.verbose
    if options.profile != None:
        profile_enabled = options.profile
        Global.config['profiling']= options.profile

    debug_build = options.debug # debug build
    clean = options.clean # enforce rebuild

    if populations is None: # Default network
        populations = Global._network[net_id]['populations']

    if projections is None: # Default network
        projections = Global._network[net_id]['projections']

    # Compiling directory
    annarchy_dir = os.getcwd() + '/' + directory

    # Turn OMP off for MacOS
    if (Global.config['paradigm']=="openmp" and Global.config['num_threads']>1 and sys.platform == "darwin"):
        Global._warning("OpenMP is not supported on Mac OS yet")
        Global.config['num_threads'] = 1

    # Test if the current ANNarchy version is newer than what was used to create the subfolder
    from pkg_resources import parse_version
    if os.path.isfile(annarchy_dir+'/release'):
        with open(annarchy_dir+'/release', 'r') as f:
            prev_release = f.read().strip()
            if parse_version(prev_release) < parse_version(ANNarchy.__release__):
                Global._print('ANNarchy has been updated, recompiling...')
                clean = True
    else:
        clean = True

    # Manage the compilation subfolder
    _folder_management(annarchy_dir, profile_enabled, clean, net_id)

    # Create a Compiler object
    compiler = Compiler(    annarchy_dir=annarchy_dir,
                            clean=clean,
                            compiler=compiler,
                            compiler_flags=compiler_flags,
                            silent=silent,
                            cuda_config=cuda_config,
                            debug_build=debug_build,
                            profile_enabled=profile_enabled,
                            populations=populations,
                            projections=projections,
                            net_id=net_id  )
    compiler.generate()

class Compiler(object):
    " Main class to generate C++ code efficiently"

    def __init__(self, annarchy_dir, clean, compiler, compiler_flags, silent, cuda_config, debug_build, profile_enabled,
                 populations, projections, net_id):

        # Store arguments
        self.annarchy_dir = annarchy_dir
        self.clean = clean
        self.compiler = compiler
        self.compiler_flags = compiler_flags
        self.silent = silent
        self.cuda_config = cuda_config
        self.debug_build = debug_build
        self.profile_enabled = profile_enabled
        self.populations = populations
        self.projections = projections
        self.net_id = net_id

    def generate(self):
        " Method to generate the C++ code."

        # Check that everything is allright in the structure of the network.
        self.check_structure()

        # Generate the code
        self.code_generation()

        # Generate the Makefile
        self.generate_makefile()

        # Copy the files if needed
        changed = self.copy_files()

        # Perform compilation if something has changed
        if changed or not os.path.isfile(self.annarchy_dir+'/ANNarchyCore'+str(self.net_id)+'.so'):
            self.compilation()

        Global._network[self.net_id]['compiled'] = True

        # Create the Python objects
        _instantiate(self.net_id)

    def copy_files(self):
        " Copy the generated files in the build/ folder if needed."
        changed = False
        if self.clean:
            for f in os.listdir(self.annarchy_dir+'/generate/net'+ str(self.net_id)):

                shutil.copy(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + f, # src
                                self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + f # dest
                    )
            changed = True

        else: # only the ones which have changed
            import filecmp
            for f in os.listdir(self.annarchy_dir+'/generate/net'+ str(self.net_id)):

                if not os.path.isfile(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + f) or \
                    not filecmp.cmp( self.annarchy_dir+'/generate/net' + str(self.net_id) + '/' + f,
                                    self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + f) :

                    shutil.copy(self.annarchy_dir+'/generate//net'+ str(self.net_id) + '/' + f, # src
                                self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' +f # dest
                    )
                    changed = True

            # Needs to check now if a file existed before in build/net but not in generate anymore
            for f in os.listdir(self.annarchy_dir+'/build/net'+ str(self.net_id)):
                if f == 'Makefile':
                    continue
                basename, extension = f.split('.')
                if not extension in ['h', 'hpp', 'cpp', 'cu']: # ex: .o
                    continue
                if not os.path.isfile(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + f):
                    if f.startswith('ANNarchyCore'):
                        continue
                    os.remove(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + f)
                    if os.path.isfile(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + basename + '.o'):
                        os.remove(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + basename + '.o')
                    changed = True

        return changed

    def compilation(self):
        """ Create ANNarchyCore.so and py extensions if something has changed. """
        # STDOUT
        if not self.silent:
            msg = 'Compiling'
            if self.net_id > 0 :
                msg += ' network ' + str(self.net_id)
            msg += '...'
            Global._print(msg)
            if Global.config['show_time']:
                t0 = time.time()

        # Switch to the build directory
        os.chdir(self.annarchy_dir + '/build/net'+ str(self.net_id))

        # Start the compilation
        verbose = "> compile_stdout.log 2> compile_stderr.log" if not Global.config["verbose"] else ""

        # Start the compilation process
        make_process = subprocess.Popen("make all -j4" + verbose, shell=True)

        # Check for errors
        if make_process.wait() != 0:
            with open('compile_stderr.log', 'r') as rfile:
                msg = rfile.read()
            Global._print(msg)
            try:
                os.remove('ANNarchyCore'+str(self.net_id)+'.so')
            except:
                pass
            Global._error('Compilation failed.')

        # Return to the current directory
        os.chdir('../../..')

        if not self.silent:
            Global._print('OK')
            if Global.config['show_time']:
                Global._print('Compilation took', time.time() - t0, 'seconds.')

    def generate_makefile(self):
        "Generate the Makefile."
        # Compiler
        if sys.platform.startswith('linux'): # Linux systems
            if self.compiler == "default":
                self.compiler = "g++"
        elif sys.platform == "darwin":   # mac os
            if self.compiler == "default":
                self.compiler = "clang++"

        # flags are common to all platforms
        if not self.debug_build:
            cpu_flags = "-march=native -O2"
        else:
            cpu_flags = "-O0 -g -D_DEBUG"

        if self.profile_enabled:
            cpu_flags += " -g"
            extra_libs.append("-lpapi")

        # OpenMP flag
        omp_flag = ""
        if Global.config['paradigm']=="openmp" and Global.config['num_threads']>1 and sys.platform != "darwin":
            omp_flag = "-fopenmp"

        # Cuda capability
        #
        # hdin (22.03.2016): we should verify in the future, if compute_35 remains as best
        # configuration for Keplar and upwards.
        cuda_gen = ""
        gpu_flags = ""
        if sys.platform.startswith('linux') and Global.config['paradigm'] == "cuda":
            from .CudaCheck import CudaCheck
            cu_version = CudaCheck().version()
            if cu_version >= (3,0):
                cuda_gen = "-gencode arch=compute_35,code=compute_35"
            else:
                cuda_gen = "-gencode arch=compute_20,code=compute_20"
            if self.debug_build:
                gpu_flags = "-g -G -D_DEBUG"

        # Extra libs from extensions such as opencv
        libs = ""
        for l in extra_libs:
            libs += str(l) + ' '

        # Python version
        py_version = "%(major)s.%(minor)s" % { 'major': sys.version_info[0],
                                               'minor': sys.version_info[1] }
        py_major = str(sys.version_info[0])

        # Python includes and libs
        # non-standard python installs need to tell the location of libpythonx.y.so/dylib
        # export LD_LIBRARY_PATH=$HOME/anaconda/lib:$LD_LIBRARY_PATH
        # export DYLD_FALLBACK_LIBRARY_PATH=$HOME/anaconda/lib:$DYLD_FALLBACK_LIBRARY_PATH
        py_prefix = sys.prefix
        if py_major=='2':
            major='2'
            test = subprocess.Popen(py_prefix + "/bin/python2-config --includes > /dev/null 2> /dev/null", shell=True)
            if test.wait()!=0:
                major=""
        else:
            major='3'
            test = subprocess.Popen(py_prefix + "/bin/python3-config --includes > /dev/null 2> /dev/null", shell=True)
            if test.wait()!=0:
                major=""

        # Test that it exists (virtualenv)
        test = subprocess.Popen("%(py_prefix)s/bin/python%(major)s-config --includes > /dev/null 2> /dev/null" % {'major': major, 'py_prefix': py_prefix}, shell=True)
        if test.wait()!=0:
            Global._warning("can not find python-config in the same directory as python, trying with the default path...")
            python_include = "`python%(major)s-config --includes`" % {'major': major, 'py_prefix': py_prefix}
            python_lib = "`python%(major)s-config --ldflags --libs`" % {'major': major, 'py_prefix': py_prefix}
        else:
            python_include = "`%(py_prefix)s/bin/python%(major)s-config --includes`" % {'major': major, 'py_prefix': py_prefix}
            python_lib = "`%(py_prefix)s/bin/python%(major)s-config --ldflags --libs`" % {'major': major, 'py_prefix': py_prefix}

        # Include path to Numpy is not standard on all distributions
        numpy_include = np.get_include()

        # Gather all Makefile flags
        makefile_flags = {
            'compiler': self.compiler,
            'cpu_flags': cpu_flags,
            'cuda_gen': cuda_gen,
            'gpu_flags': gpu_flags,
            'openmp': omp_flag,
            'libs': libs,
            'py_version': py_version,
            'py_major': py_major,
            'python_include': python_include,
            'python_lib': python_lib,
            'numpy_include': numpy_include,
            'net_id': self.net_id
        }

        # Create Makefiles depending on the target platform and parallel framework
        if sys.platform.startswith('linux'): # Linux systems
            if Global.config['paradigm'] == "cuda":
                makefile_template = linux_cuda_template
            else:
                makefile_template = linux_omp_template

        elif sys.platform == "darwin":   # mac os
            makefile_template = osx_seq_template

        else: # Windows: to test....
            Global._warning("Compilation on windows is not supported yet.")


        # Write the Makefile to the disk
        with open(self.annarchy_dir + '/generate/net'+ str(self.net_id) + '/Makefile', 'w') as wfile:
            wfile.write(makefile_template % makefile_flags)


    def code_generation(self):
        """ Code generation dependent on paradigm """
        from .CodeGenerator import CodeGenerator
        generator = CodeGenerator(self.annarchy_dir, self.populations, self.projections, self.net_id)
        generator.generate()

    def check_structure(self):
        """
        Checks the structure to display more useful error messages.
        """
        # Check populations
        for pop in self.populations:
            # Reserved variable names
            for term in ['t', 'dt', 't_pre', 't_post', 't_spike']:
                if term in pop.attributes:
                    Global._print(pop.neuron_type.parameters)
                    Global._print(pop.neuron_type.equations)
                    Global._error(term + ' is a reserved variable name')


        # Check projections
        for proj in self.projections:
            # Reserved variable names
            for term in ['t', 'dt', 't_pre', 't_post', 't_spike']:
                if term in proj.attributes:
                    Global._print(proj.synapse_type.parameters)
                    Global._print(proj.synapse_type.equations)
                    Global._error(term + ' is a reserved variable name')

            # Check the connector method has been called
            if not proj._connection_method:
                Global._error('The projection between populations', proj.pre.id, 'and', proj.post.id, 'has not been connected. Call a connector method before compiling the network.')


            # Check existing pre variables
            for dep in  proj.synapse_type.description['dependencies']['pre']:
                if dep.startswith('sum('):
                    target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                    if not target in proj.pre.targets:
                        Global._error('The pre-synaptic population ' + proj.pre.name + ' receives no projection with the type ' + target)
                    continue

                if not dep in proj.pre.attributes:
                    Global._error('The pre-synaptic population ' + proj.pre.name + ' has no variable called ' + dep)

            for dep in  proj.synapse_type.description['dependencies']['post']:
                if dep.startswith('sum('):
                    target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                    if not target in proj.post.targets:
                        Global._error('The post-synaptic population ' + proj.post.name + ' receives no projection with the type ' + target)
                    continue

                if not dep in proj.post.attributes:
                    Global._error('The post-synaptic population ' + proj.post.name + ' has no variable called ' + dep)


def _instantiate(net_id, import_id=-1):
    """ After every is compiled, actually create the Cython objects and
        bind them to the Python ones."""

    if import_id < 0:
        import_id = net_id

    if Global.config['verbose']:
        Global._print('Building network ...')

    # Import the Cython library
    try:
        cython_module = imp.load_dynamic('ANNarchyCore'+str(import_id), 'annarchy/ANNarchyCore'+str(import_id)+'.so')
    except Exception as e:
        Global._print(e)
        Global._error('Something went wrong when importing the network. Force recompilation with --clean.')

    Global._network[net_id]['instance'] = cython_module

    # Bind the py extensions to the corresponding python objects
    for pop in Global._network[net_id]['populations']:
        if Global.config['verbose']:
            Global._print('Creating population', pop.name)
        if Global.config['show_time']:
            t0 = time.time()

        # Instantiate the population
        pop._instantiate(cython_module)

        if Global.config['show_time']:
            Global._print('Creating', pop.name, 'took', (time.time()-t0)*1000, 'milliseconds')

    # Instantiate projections
    for proj in Global._network[net_id]['projections']:
        if Global.config['verbose']:
            Global._print('Creating projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')
        if Global.config['show_time']:
            t0 = time.time()

        # Create the projection
        proj._instantiate(cython_module)

        if Global.config['show_time']:
            Global._print('Creating the projection took', (time.time()-t0)*1000, 'milliseconds')

    # Finish to initialize the network, especially the rng
    # Must be called after the pops and projs are created!
    cython_module.pyx_create(Global.config['dt'], Global.config['seed'])

    # Transfer initial values
    for pop in Global._network[net_id]['populations']:
        if Global.config['verbose']:
            Global._print('Initializing population', pop.name)
        pop._init_attributes()
    for proj in Global._network[net_id]['projections']:
        if Global.config['verbose']:
            Global._print('Initializing projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')
        proj._init_attributes()

    # Sets the desired number of threads
    if Global.config['num_threads'] > 1:
        cython_module.set_number_threads(Global.config['num_threads'])

    # Start the monitors
    for monitor in Global._network[net_id]['monitors']:
        monitor._init_monitoring()
