"""

    Generator.py
    
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

from optparse import OptionParser
from optparse import OptionGroup

# String containing the extra libs which can be added by extensions
# e.g. extra_libs = ['-lopencv_core', '-lopencv_video']
extra_libs = []
 
def _folder_management(annarchy_dir, profile_enabled, clean):
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

    if not os.path.exists(annarchy_dir):    
        os.mkdir(annarchy_dir)
        os.mkdir(annarchy_dir+'/build')
        
    # Create the generate subfolder
    shutil.rmtree(annarchy_dir+'/generate', True)
    os.mkdir(annarchy_dir+'/generate')

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

def compile(directory='annarchy', clean=False, populations=None, projections=None, silent=False, cpp_stand_alone=False, debug_build=False, profile_enabled = False, net_id=0):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will perform the simulation.
    
    *Parameters*:

    * **directory**: name of the subdirectory where the code will be generated and compiled.
    * **clean**: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * **populations**: list of populations which should be compiled. If set to None, all available populations will be used.
    * **projections**: list of projection which should be compiled. If set to None, all available projections will be used.
    * **projections**: list of projection which should be compiled. If set to None, all available projections will be used.

    The following arguments are for internal use only:

    * **cpp_stand_alone**: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
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

    if populations == None: # Default network
        populations = Global._network[net_id]['populations']

    if projections == None: # Default network
        projections = Global._network[net_id]['projections']

    # Compiling directory
    annarchy_dir = os.getcwd() + '/' + directory

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
    _folder_management(annarchy_dir, profile_enabled, clean)
    
    # Create a Generator object
    generator = Generator(annarchy_dir, clean, silent, cpp_stand_alone, debug_build, profile_enabled, 
                 populations, projections, net_id)
    generator.generate()
    
class Generator(object):
    " Main class to generate C++ code efficiently"
      
    def __init__(self, annarchy_dir, clean, silent, cpp_stand_alone, debug_build, profile_enabled, 
                 populations, projections, net_id): 
        
        # Store arguments
        self.annarchy_dir = annarchy_dir
        self.clean = clean
        self.silent = silent
        self.cpp_stand_alone = cpp_stand_alone
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
        self.code_generation(self.cpp_stand_alone, self.profile_enabled, 
                                                        self.clean)
        # Copy the files if needed
        changed = self.copy_files(self.clean)
        
        # Perform compilation if something has changed
        if changed or not os.path.isfile(self.annarchy_dir+'/ANNarchyCore'+str(self.net_id)+'.so'):
            self.compilation()

            # Return to the current directory
            os.chdir('..')
                
        Global._network[self.net_id]['compiled'] = True
        
        # Create the Python objects                
        _instantiate(self.net_id)    

    def copy_files(self, clean):
        " Copy the generated files in the build/ folder if needed."
        changed = False
        if clean:
            for file in os.listdir(self.annarchy_dir+'/generate'):
                shutil.copy(self.annarchy_dir+'/generate/'+file, # src
                            self.annarchy_dir+'/build/'+file # dest
                )
            changed = True
        else: # only the ones which have changed
            import filecmp
            for file in os.listdir(self.annarchy_dir+'/generate'):
                if  not os.path.isfile(self.annarchy_dir+'/build/'+file) or \
                    not filecmp.cmp( self.annarchy_dir+'/generate/'+file, 
                                    self.annarchy_dir+'/build/'+file) :
                    shutil.copy(self.annarchy_dir+'/generate/'+file, # src
                                self.annarchy_dir+'/build/'+file # dest
                    )
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

        # flags are common to all platforms
        if not self.debug_build:
            cpu_flags = "-march=native -O2"
            gpu_flags = ""
        else:
            cpu_flags = "-O0 -g -D_DEBUG"
            gpu_flags = ""

        if self.profile_enabled:
            cpu_flags += " -g"
            extra_libs.append("-lpapi")

        # Extra libs from extensions such as opencv
        libs = ""
        for l in extra_libs:
            libs += str(l) + ' '

        # Python version
        py_version = "%(major)s.%(minor)s" % { 'major': sys.version_info[0],
                                               'minor': sys.version_info[1] }
        py_major = str(sys.version_info[0])

        # Include path to Numpy is not standard on all distributions
        numpy_include = np.get_include()

        # switch to build-directory
        os.chdir(self.annarchy_dir)

        # create Makefiles dependent on target platform
        if sys.platform.startswith('linux'): # Linux systems
            src = """# Makefile generated by ANNarchy
seq:
\tcython -%(major)s build/ANNarchyCore%(net_id)s.pyx --cplus
\tg++ -march=native %(cpu_flags)s -shared -fPIC -fpermissive -std=c++0x -I. -I/usr/include/python%(py_version)s -I/usr/include/python%(py_version)sm -I %(numpy_include)s build/*.cpp %(libs)s -o ANNarchyCore%(net_id)s.so

omp:
\tcython -%(major)s build/ANNarchyCore%(net_id)s.pyx --cplus
\tg++ -march=native %(cpu_flags)s -shared -fPIC -fpermissive -std=c++0x -I. -I/usr/include/python%(py_version)s -I/usr/include/python%(py_version)sm -I %(numpy_include)s -fopenmp build/*.cpp %(libs)s -o ANNarchyCore%(net_id)s.so

# CC 2.0 (Tesla)
cuda_20:
\tcython -%(major)s build/ANNarchyCore%(net_id)s.pyx --cplus
\tnvcc -gencode arch=compute_20,code=compute_20 %(gpu_flags)s -I/usr/include/python2.7 build/*.cu build/*.cpp -lpython2.7 -Xcompiler -fPIC -shared -o ANNarchyCore%(net_id)s.so

# CC 3.5 (Keplar)
cuda_35:
\tcython -%(major)s build/ANNarchyCore%(net_id)s.pyx --cplus
\tnvcc -gencode arch=compute_35,code=compute_35 %(gpu_flags)s -I/usr/include/python2.7 build/*.cu build/*.cpp -lpython2.7 -Xcompiler -fPIC -shared -o ANNarchyCore%(net_id)s.so

clean:
\trm -rf build/*.o
\trm -rf build/*.so
""" % {'cpu_flags': cpu_flags, 'gpu_flags': gpu_flags, 'libs': libs, 'py_version': py_version, 'numpy_include': numpy_include, 'major': py_major, 'net_id': self.net_id}

        elif sys.platform == "darwin":   # mac os
            src = """# Makefile generated by ANNarchy
seq:
\tcython -%(major)s build/ANNarchyCore%(net_id)s.pyx --cplus
\tclang++ -stdlib=libc++ -std=c++11 -fPIC -shared %(cpu_flags)s -fpermissive -I. -I/usr/include/python%(py_version)s -I/usr/include/python%(py_version)sm -I %(numpy_include)s build/*.cpp -lpython %(libs)s -o ANNarchyCore%(net_id)s.so

clean:
\trm -rf build/*.o
\trm -rf build/*.so
""" % {'cpu_flags': cpu_flags, 'libs': libs, 'py_version': py_version, 'numpy_include': numpy_include, 'net_id': self.net_id}

        else: # Windows: to test....
            Global._warning("Compilation on windows is not supported yet.")
            exit(0)

        #
        # Write the Makefile to the disk
        with open('Makefile', 'w') as wfile:
            wfile.write(src)

        #
        # Start the compilation
        verbose = "> compile_stdout.log 2> compile_stderr.log" if not Global.config["verbose"] else ""

        if Global.config['paradigm'] == "cuda":
            from .CudaCheck import CudaCheck
            cu_version = CudaCheck().version()

            if cu_version >= (3,0):
                Global._print("Using CUDA 3.x")
                make_process = subprocess.Popen("make cuda_35 -j4 "+ verbose, shell=True)
            else:
                Global._print("Using CUDA 2.x")
                make_process = subprocess.Popen("make cuda_20 -j4 "+ verbose, shell=True)

        elif (Global.config['paradigm']=="openmp" and Global.config['num_threads']>1):
        
            if sys.platform == "darwin":
                Global._warning("OpenMP is not supported on Mac OS yet")
                exit(0)
            else:
                make_process = subprocess.Popen("make omp " + verbose, shell=True)

        else:
            make_process = subprocess.Popen("make seq " + verbose, shell=True)

        if make_process.wait() != 0:
            with open('compile_stderr.log', 'r') as rfile:
                msg = rfile.read()
            Global._print(msg)
            Global._error('Compilation failed.')
            try:
                os.remove('ANNarchyCore'+self.net_id+'.so')
            except:
                pass
            exit(0)

        if not self.silent:
            Global._print('OK')
            if Global.config['show_time']:
                Global._print('Compilation took', time.time() - t0, 'seconds.')


    def code_generation(self, cpp_stand_alone, profile_enabled, clean):
        """ Code generation dependent on paradigm """
        if Global.config['paradigm'] == "openmp":
            from .OMP.OMPGenerator import OMPGenerator
            generator = OMPGenerator(self.annarchy_dir, self.populations, self.projections, self.net_id)
            generator.generate()
        else: # CUDA
            from .CUDA.CUDAGenerator import CUDAGenerator
            generator = CUDAGenerator(self.annarchy_dir, self.populations, self.projections, self.net_id)
            generator.generate()

        if self.profile_enabled:
            from .Profile.ProfileGenerator import ProfileGenerator
            prof_generator = ProfileGenerator(self.annarchy_dir, self.populations, self.projections)
            prof_generator.generate()

    def check_structure(self):
        """
        Checks the structure to display more useful error messages.
        """
        # Check populations
        for pop in self.populations:
            # Reserved variable names
            for term in ['t', 'dt', 't_pre', 't_post']:
                if term in pop.attributes:
                    Global._print(pop.neuron_type.parameters)
                    Global._print(pop.neuron_type.equations)
                    Global._error(term + ' is a reserved variable name')
                    exit(0)

        # Check projections
        for proj in self.projections:
            # Reserved variable names
            for term in ['t', 'dt', 't_pre', 't_post']:
                if term in proj.attributes:
                    Global._print(proj.synapse.parameters)
                    Global._print(proj.synapse.equations)
                    Global._error(term + ' is a reserved variable name')
                    exit(0)
            # Check the connector method has been called
            if not proj._synapses:
                Global._error('The projection between populations', proj.pre.id, 'and', proj.post.id, 'has not been connected. Call a connector method before compiling the network.')
                exit(0)

            # Check existing pre variables
            for dep in  proj.synapse.description['dependencies']['pre']:
                if dep.startswith('sum('):
                    target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                    if not target in proj.pre.targets:
                        Global._error('The pre-synaptic population ' + proj.pre.name + ' receives no projection with the type ' + target)
                        exit(0)
                    continue 
                if not dep in proj.pre.attributes:
                    Global._error('The pre-synaptic population ' + proj.pre.name + ' has no variable called ' + dep)
                    exit(0)
            for dep in  proj.synapse.description['dependencies']['post']:
                if dep.startswith('sum('):
                    target = re.findall(r'\(([\s\w]+)\)', dep)[0].strip()
                    if not target in proj.post.targets:
                        Global._error('The post-synaptic population ' + proj.post.name + ' receives no projection with the type ' + target)
                        exit(0)
                    continue 
                if not dep in proj.post.attributes:
                    Global._error('The post-synaptic population ' + proj.post.name + ' has no variable called ' + dep)
                    exit(0)


def _instantiate(net_id, single=True):
    """ After every is compiled, actually create the Cython objects and 
        bind them to the Python ones."""

    if Global.config['verbose']:
        Global._print('Building network ...')

    # Import the Cython library
    try:
        #cython_module = __import__('ANNarchyCore'+str(base_net))
        if single:
            cython_module = imp.load_dynamic('ANNarchyCore'+str(net_id), 'annarchy/ANNarchyCore'+str(net_id)+'.so')
        else:
            sys.path.append('annarchy/net'+str(net_id))
            cython_module = imp.load_dynamic('ANNarchyCore0', 'annarchy/net'+str(net_id)+'/ANNarchyCore0.so')
    except Exception as e:
        Global._print(e)
        Global._error('Something went wrong when importing the network. Force recompilation with --clean.')
        exit(0)
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