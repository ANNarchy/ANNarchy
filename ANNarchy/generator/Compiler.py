#===============================================================================
#
#     Compiler.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
import os, sys, imp
import subprocess
import shutil
import time
import re
import json
import argparse
import numpy as np

# ANNarchy core informations
import ANNarchy
import ANNarchy.core.Global as Global
from ANNarchy.extensions.bold.NormProjection import _update_num_aff_connections
from .Template.MakefileTemplate import *
from .Sanity import check_structure

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

    # Save current ANNarchy version and paradigm
    with open(annarchy_dir+'/release', 'w') as wfile:
        wfile.write(Global.config['paradigm']+', '+ANNarchy.__release__)

    sys.path.append(annarchy_dir)

def setup_parser():
    """
    ANNarchy scripts can be run by several command line arguments. These are
    checked with the ArgumentParser provided by Python.
    """
    # override the error behavior of OptionParser,
    # normally an unknwon arg would raise an exception
    parser = argparse.ArgumentParser(description='ANNarchy: Artificial Neural Networks architect.')

    group = parser.add_argument_group('General')
    group.add_argument("-c", "--clean", help="Forces recompilation.", action="store_true", default=False, dest="clean")
    group.add_argument("-d", "--debug", help="Compilation with debug symbols and additional checks.", action="store_true", default=False, dest="debug")
    group.add_argument("-v", "--verbose", help="Shows all messages.", action="store_true", default=None, dest="verbose")
    group.add_argument("--prec", help="Set the floating precision used.", action="store", type=str, default=None, dest="precision")

    group = parser.add_argument_group('OpenMP')
    group.add_argument("-j", "--num_threads", help="Number of threads to use.", type=int, action="store", default=None, dest="num_threads")

    group = parser.add_argument_group('CUDA')
    group.add_argument("--gpu", help="Enables CUDA and optionally specifies the GPU id (default: 0).", type=int, action="store", nargs='?', default=-1, const=0, dest="gpu_device")


    group = parser.add_argument_group('Internal')
    group.add_argument("--profile", help="Enables profiling.", action="store_true", default=None, dest="profile")
    group.add_argument("--profile_out", help="Target file for profiling data.", action="store", type=str, default=None, dest="profile_out")


    return parser

def compile(
        directory='annarchy',
        clean=False,
        populations=None,
        projections=None,
        compiler="default",
        compiler_flags="default",
        cuda_config={'device': 0},
        annarchy_json="",
        silent=False,
        debug_build=False,
        profile_enabled=False,
        net_id=0
    ):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will perform the simulation.

    The ``compiler``, ``compiler_flags`` and part of ``cuda_config`` take their default value from the configuration file ``~/.config/ANNarchy/annarchy.json``.

    The following arguments are for internal development use only:

    * **debug_build**: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    * **profile_enabled**: creates a profilable version of ANNarchy, which logs several computation timings (default: False).
    
    :param directory: name of the subdirectory where the code will be generated and compiled. Must be a relative path. Default: "annarchy/".
    :param clean: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    :param populations: list of populations which should be compiled. If set to None, all available populations will be used.
    :param projections: list of projection which should be compiled. If set to None, all available projections will be used.
    :param compiler: C++ compiler to use. Default: g++ on GNU/Linux, clang++ on OS X. Valid compilers are [g++, clang++].
    :param compiler_flags: platform-specific flags to pass to the compiler. Default: "-march=native -O2". Warning: -O3 often generates slower code and can cause linking problems, so it is not recommended.
    :param cuda_config: dictionary defining the CUDA configuration for each population and projection.
    :param annarchy_json: compiler flags etc can be stored in a .json file normally placed in the home directory (see comment below). With this flag one can directly assign a file location.
    :param silent: defines if the "Compiling... OK" should be printed.

    """
    # Check if the network has already been compiled
    if Global._network[net_id]['compiled']:
        Global._print("""compile(): the network has already been compiled, doing nothing.
    If you are re-running a Jupyter notebook, you should call `clear()` right after importing ANNarchy in order to reset everything.""")
        return

    # Get the command-line arguments
    parser = setup_parser()
    options, unknown = parser.parse_known_args()
    #if len(unknown) > 0:
    #    Global._warning('unrecognized arguments:', unknown)

    # if the parameters set on command-line they overwrite Global.config
    if options.num_threads != None:
        Global.config['num_threads'] = options.num_threads

    # Get CUDA configuration
    if options.gpu_device >= 0:
        Global.config['paradigm'] = "cuda"
        cuda_config['device'] = int(options.gpu_device)

    # Check that CUDA is enabled
    try:
        from .CudaCheck import CudaCheck
    except:
        Global._error('CUDA is not installed on your system')

    # Check that a single backend is chosen
    if (options.num_threads != None) and (options.gpu_device >= 0):
        Global._error('CUDA and openMP can not be active at the same time, please check your command line arguments.')

    # Verbose
    if options.verbose != None:
        Global.config['verbose'] = options.verbose

    # Precision
    if options.precision != None:
        Global.config['precision'] = options.precision

    # Profiling
    if options.profile != None:
        profile_enabled = options.profile
        Global.config['profiling'] = options.profile
        Global.config['profile_out'] = options.profile_out
    if profile_enabled != False and options.profile == None:
        # Profiling enabled due compile()
        Global.config['profiling'] = True

    # Debug
    if not debug_build:
        debug_build = options.debug  # debug build
    Global.config["debug"] = debug_build

    # Clean
    clean = options.clean or clean # enforce rebuild

    # Populations to compile
    if populations is None: # Default network
        populations = Global._network[net_id]['populations']

    # Projections to compile
    if projections is None: # Default network
        projections = Global._network[net_id]['projections']

    # Compiling directory
    annarchy_dir = os.getcwd() + '/' + directory
    if not annarchy_dir.endswith('/'):
        annarchy_dir += '/'
    Global._network[net_id]['directory'] = annarchy_dir

    # Turn OMP off for MacOS
    if (Global._check_paradigm("openmp") and Global.config['num_threads'] > 1 and sys.platform == "darwin"):
        Global._warning("OpenMP is not supported on Mac OS yet")
        Global.config['num_threads'] = 1

    # Test if the current ANNarchy version is newer than what was used to create the subfolder
    from pkg_resources import parse_version
    if os.path.isfile(annarchy_dir+'/release'):
        with open(annarchy_dir+'/release', 'r') as rfile:
            prev_release = rfile.read().strip()
            prev_paradigm = ''

            # HD (03.08.2016):
            # in ANNarchy 4.5.7b I added also the paradigm to the release tag.
            # This if clause can be removed in later releases (TODO)
            if prev_release.find(',') != -1:
                prev_paradigm, prev_release = prev_release.split(', ')
            else:
                # old release tag
                clean = True

            if parse_version(prev_release) < parse_version(ANNarchy.__release__):
                clean = True

            elif prev_paradigm != Global.config['paradigm']:
                clean = True

    else:
        clean = True # for very old versions

    # Check if the last compilation was successful
    if os.path.isfile(annarchy_dir+'/compilation'):
        with open(annarchy_dir + '/compilation', 'r') as rfile:
            res = rfile.read()
            if res.strip() == "0": # the last compilation failed
                clean = True
    else:
        clean = True

    # Manage the compilation subfolder
    _folder_management(annarchy_dir, profile_enabled, clean, net_id)

    # Create a Compiler object
    compiler = Compiler(
        annarchy_dir=annarchy_dir,
        clean=clean,
        compiler=compiler,
        compiler_flags=compiler_flags,
        path_to_json=annarchy_json,
        silent=silent,
        cuda_config=cuda_config,
        debug_build=debug_build,
        profile_enabled=profile_enabled,
        populations=populations,
        projections=projections,
        net_id=net_id
    )

    # Code Generation
    compiler.generate()

    # Create the Python objects
    _instantiate( compiler.net_id, cuda_config=compiler.cuda_config, user_config=compiler.user_config)

    # NormProjections require an update of afferent projections
    _update_num_aff_connections(compiler.net_id)

def python_environment():
    """
    Python environment configuration, required by Compiler.generate_makefile. Contains among others the python version, library path and cython version.

    Warning: changes to this method should be copied to setup.py.
    """
    # Python version
    py_version = "%(major)s.%(minor)s" % {'major': sys.version_info[0],
                                          'minor': sys.version_info[1]}
    py_major = str(sys.version_info[0])

    if py_major == '2':
        Global._warning("Python 2 is not supported anymore, things might break.")

    # Python includes and libs
    # non-standard python installs need to tell the location of libpythonx.y.so/dylib
    # export LD_LIBRARY_PATH=$HOME/anaconda/lib:$LD_LIBRARY_PATH
    # export DYLD_FALLBACK_LIBRARY_PATH=$HOME/anaconda/lib:$DYLD_FALLBACK_LIBRARY_PATH
    py_prefix = sys.prefix

    # Search for pythonx.y-config
    cmd = "%(py_prefix)s/bin/python%(py_version)s-config --includes > /dev/null 2> /dev/null"
    with subprocess.Popen(cmd % {'py_version': py_version, 'py_prefix': py_prefix}, shell=True) as test:
        if test.wait() != 0:
            Global._warning("Can not find python-config in the same directory as python, trying with the default path...")
            python_config_path = "python%(py_version)s-config"% {'py_version': py_version}
        else:
            python_config_path = "%(py_prefix)s/bin/python%(py_version)s-config" % {'py_version': py_version, 'py_prefix': py_prefix}

    python_include = "`%(pythonconfigpath)s --includes`" % {'pythonconfigpath': python_config_path}
    python_libpath = "-L%(py_prefix)s/lib" % {'py_prefix': py_prefix}

    # Identify the -lpython flag
    with subprocess.Popen('%(pythonconfigpath)s --ldflags' % {'pythonconfigpath': python_config_path},
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as test:
        flagline = str(test.stdout.read().decode('UTF-8')).strip()
        errorline = str(test.stderr.read().decode('UTF-8'))
        test.wait()

    if len(errorline) > 0:
        Global._error("Unable to find python-config. Make sure you have installed the development files of Python (python-dev or -devel) and that either python-config, python2-config or python3-config are in your path.")
    flags = flagline.split(' ')
    for flag in flags:
        if flag.startswith('-lpython'):
            python_lib = flag
            break
    else:
        python_lib = "-lpython" + py_version

    # Check cython version
    with subprocess.Popen(py_prefix + "/bin/cython%(major)s -V > /dev/null 2> /dev/null" % {'major': py_major}, shell=True) as test:
        if test.wait() != 0:
            cython = py_prefix + "/bin/cython"
        else:
            cython = py_prefix + "/bin/cython" + py_major
    # If not in the same folder as python, use the default
    with subprocess.Popen("%(cython)s -V > /dev/null 2> /dev/null" % {'cython': cython}, shell=True) as test:
        if test.wait() != 0:
            cython = shutil.which("cython"+str(py_major))
            if cython is None:
                cython = shutil.which("cython")
                if cython is None:
                    Global._error("Unable to detect the path to cython." )

    return py_version, py_major, python_include, python_lib, python_libpath, cython

class Compiler(object):
    " Main class to generate C++ code efficiently"

    def __init__(self, annarchy_dir, clean, compiler, compiler_flags, path_to_json, silent, cuda_config, debug_build, profile_enabled,
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

        # Get user-defined config
        self.user_config = {
            'openmp': {
                'compiler': 'clang++' if sys.platform == "darwin" else 'g++',
                'flags' : "-march=native -O2",
            }
        }

        if len(path_to_json) == 0:
            # check homedirectory
            if os.path.exists(os.path.expanduser('~/.config/ANNarchy/annarchy.json')):
                with open(os.path.expanduser('~/.config/ANNarchy/annarchy.json'), 'r') as rfile:
                    self.user_config = json.load(rfile)
        else:
            with open(path_to_json, 'r') as rfile:
                self.user_config = json.load(rfile)

    def generate(self):
        "Method to generate the C++ code."
        if Global._profiler:
            t0 = time.time()

        # Check that everything is allright in the structure of the network.
        check_structure(self.populations, self.projections)

        # Generate the code
        self.code_generation()

        # Generate the Makefile
        self.generate_makefile()

        # Copy the files if needed
        changed = self.copy_files()

        # Perform compilation if something has changed
        if changed or not os.path.isfile(self.annarchy_dir + '/ANNarchyCore' + str(self.net_id) + '.so'):
            self.compilation()

        Global._network[self.net_id]['compiled'] = True
        if Global._profiler:
            t1 = time.time()
            Global._profiler.add_entry( t0, t1, "compile()", "compile")

    def copy_files(self):
        " Copy the generated files in the build/ folder if needed."
        changed = False
        if self.clean:
            for file in os.listdir(self.annarchy_dir+'/generate/net'+ str(self.net_id)):
                if file.endswith(".log"):
                    continue

                shutil.copy(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + file, # src
                            self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + file # dest
                           )
            changed = True

        else: # only the ones which have changed
            import filecmp
            for file in os.listdir(self.annarchy_dir+'/generate/net'+ str(self.net_id)):
                if file.endswith(".log"):
                    continue

                if not os.path.isfile(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + file) or \
                    not filecmp.cmp(self.annarchy_dir+'/generate/net' + str(self.net_id) + '/' + file,
                                    self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + file):


                    shutil.copy(self.annarchy_dir+'/generate//net'+ str(self.net_id) + '/' + file, # src
                                self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' +file # dest
                               )
                    changed = True
                    # For debugging
                    # print(f, 'has changed')
                    # with open(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + f, 'r') as rfile:
                    #     text = rfile.read()
                    #     print(text)

            # Needs to check now if a file existed before in build/net but not in generate anymore
            for file in os.listdir(self.annarchy_dir+'/build/net'+ str(self.net_id)):
                if file == 'Makefile':
                    continue
                if file.endswith(".log"):
                    continue
                basename, extension = os.path.splitext(file)
                if not extension in ['h', 'hpp', 'cpp', 'cu']: # ex: .o
                    continue
                if not os.path.isfile(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + file):
                    if file.startswith('ANNarchyCore'):
                        continue
                    os.remove(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + file)
                    if os.path.isfile(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + basename + '.o'):
                        os.remove(self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + basename + '.o')
                    changed = True

        return changed

    def compilation(self):
        """ Create ANNarchyCore.so and py extensions if something has changed. """
        # STDOUT
        if not self.silent:
            msg = 'Compiling'
            if self.net_id > 0:
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
            with open(self.annarchy_dir + '/compilation', 'w') as wfile:
                wfile.write("0")
            Global._print(msg)
            try:
                os.remove('ANNarchyCore'+str(self.net_id)+'.so')
            except:
                pass
            Global._error('Compilation failed.')
        else: # Note that the last compilation was successful
            with open(self.annarchy_dir + '/compilation', 'w') as wfile:
                wfile.write("1")

        # Return to the current directory
        os.chdir('../../..')

        if not self.silent:
            Global._print('OK')
            if Global.config['show_time']:
                Global._print('Compilation took', time.time() - t0, 'seconds.')

    def generate_makefile(self):
        """
        Generate the Makefile.

        The makefile consists of two stages compile the cython wrapper and
        compile the ANNarchy model files. Both is then linked together to
        a shared library usable in Python.
        """
        # Compiler
        if self.compiler == "default":
            self.compiler = self.user_config['openmp']['compiler']
        if self.compiler_flags == "default":
            self.compiler_flags = self.user_config['openmp']['flags']

        # flags are common to all platforms
        if not self.debug_build:
            cpu_flags = self.compiler_flags
        else:
            cpu_flags = "-O0 -g -D_DEBUG"

        if self.profile_enabled:
            cpu_flags += " -g"
            #extra_libs.append("-lpapi")

        # OpenMP flag
        omp_flag = ""
        if Global.config['paradigm'] == "openmp" and sys.platform != "darwin":
            omp_flag = "-fopenmp"

        # Cuda Library and Compiler
        #
        # hdin (22.03.2016): we should verify in the future, if compute_35 remains as best
        # configuration for Keplar and upwards.
        cuda_gen = ""
        gpu_flags = ""
        gpu_compiler = "nvcc"
        gpu_ldpath = ""
        if sys.platform.startswith('linux') and Global.config['paradigm'] == "cuda":
            from .CudaCheck import CudaCheck
            cu_version = CudaCheck().version_str()

            if int(cu_version) < 30:
                Global._warning("You seem to use a GPU with CC < 3.0 this might lead to problems with newer CUDA SDKs.")
            else:
                cuda_gen = "-arch sm_%(ver)s" % {'ver': cu_version}

            if self.debug_build:
                gpu_flags = "-g -G -D_DEBUG"

            # read the config file for the cuda lib path
            if 'cuda' in self.user_config.keys():
                gpu_compiler = self.user_config['cuda']['compiler']
                gpu_ldpath = '-L' + self.user_config['cuda']['path'] + '/lib'

        # Extra libs from extensions such as opencv
        libs = ""
        for lib in extra_libs:
            libs += str(lib) + ' '

        # Python environment
        py_version, py_major, python_include, python_lib, python_libpath, cython = python_environment()

        # Include path to Numpy is not standard on all distributions
        numpy_include = np.get_include()

        # The connector module needs to reload some header files,
        # ANNarchy.__path__ provides the installation directory
        path_to_cython_ext = "-I "+ANNarchy.__path__[0]+'/core/cython_ext/ -I '+ANNarchy.__path__[0][:-8]

        # Gather all Makefile flags
        makefile_flags = {
            'compiler': self.compiler,
            'cpu_flags': cpu_flags,
            'cuda_gen': cuda_gen,
            'gpu_compiler': gpu_compiler,
            'gpu_flags': gpu_flags,
            'gpu_ldpath': gpu_ldpath,
            'openmp': omp_flag,
            'extra_libs': libs,
            'py_version': py_version,
            'py_major': py_major,
            'cython': cython,
            'python_include': python_include,
            'python_lib': python_lib,
            'python_libpath': python_libpath,
            'numpy_include': numpy_include,
            'net_id': self.net_id,
            'cython_ext': path_to_cython_ext
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
        generator = CodeGenerator(self.annarchy_dir, self.populations, self.projections, self.net_id, self.cuda_config)
        generator.generate()



def _instantiate(net_id, import_id=-1, cuda_config=None, user_config=None):
    """ After every is compiled, actually create the Cython objects and
        bind them to the Python ones."""
    if Global._profiler:
        t0 = time.time()

    # parallel_run(number=x) defines multiple networks (net_id) but only network0 is compiled
    if import_id < 0:
        import_id = net_id

    # subdirectory where the library lies
    annarchy_dir = Global._network[import_id]['directory']
    libname = 'ANNarchyCore' + str(import_id)
    libpath = annarchy_dir + '/' + libname + '.so'

    if Global.config['verbose']:
        Global._print('Loading library...', libname, libpath)

    # Import the Cython library
    try:
        cython_module = imp.load_dynamic(
                libname, # Name of the network
                libpath # Path to the library
        )
    except Exception as e:
        Global._print(e)
        Global._error('Something went wrong when importing the network. Force recompilation with --clean.')

    if Global.config['verbose']:
        Global._print('Library loaded.')

    Global._network[net_id]['instance'] = cython_module

    # Set the CUDA device
    if Global._check_paradigm("cuda"):
        device = 0
        if cuda_config:
            device = int(cuda_config['device'])
        elif 'cuda' in user_config['cuda']:
            device = int(user_config['cuda']['device'])

        if Global.config['verbose']:
            Global._print('Setting GPU device', device)
        cython_module.set_device(device)

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
            Global._print('Creating projection from', proj.pre.name, 'to', proj.post.name, 'with target="', proj.target, '"')
        if Global.config['show_time']:
            t0 = time.time()

        # Create the projection
        proj._instantiate(cython_module)

        if Global.config['show_time']:
            Global._print('Creating the projection took', (time.time()-t0)*1000, 'milliseconds')

    # Finish to initialize the network
    cython_module.pyx_create(Global.config['dt'], Global.config['seed'])

    # Set the user-defined constants
    for obj in Global._objects['constants']:
        getattr(cython_module, '_set_'+obj.name)(obj.value)

    # Transfer initial values
    for pop in Global._network[net_id]['populations']:
        if Global.config['verbose']:
            Global._print('Initializing population', pop.name)
        pop._init_attributes()
    for proj in Global._network[net_id]['projections']:
        if Global.config['verbose']:
            Global._print('Initializing projection', proj.name, 'from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')
        proj._init_attributes()

    # The rng dist must be initialized after the pops and projs are created!
    if Global._check_paradigm("openmp"):
        cython_module.pyx_init_rng_dist()

    # Sets the desired number of threads
    if Global.config['num_threads'] > 1 and Global._check_paradigm("openmp"):
        cython_module.set_number_threads(Global.config['num_threads'])
        if Global.config['verbose']:
            Global._print('Running simulation with', Global.config['num_threads'], 'threads.')

    # Start the monitors
    for monitor in Global._network[net_id]['monitors']:
        monitor._init_monitoring()

    if Global._profiler:
        t1 = time.time()
        Global._profiler.add_entry( t0, t1, "instantiate()", "compile")
