"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import os, sys, importlib
import subprocess
import shutil
import multiprocessing
import time
import json
import numpy as np

# ANNarchy core informations
import ANNarchy

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern.ConfigManagement import _update_global_config, _check_paradigm, ConfigManager
from ANNarchy.intern import Messages

from ANNarchy.extensions.bold.NormProjection import _update_num_aff_connections
from ANNarchy.generator.Template.CMakeTemplate import *
from ANNarchy.generator.CodeGenerator import CodeGenerator
from ANNarchy.generator.Sanity import check_structure, check_experimental_features
from ANNarchy.generator.Utils import check_cuda_version
from ANNarchy.parser.report.Report import report

from packaging.version import parse as parse_version

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
    if ConfigManager().get('verbose', net_id):
        Messages._print("Create subdirectory.")

    if clean or profile_enabled:
        shutil.rmtree(annarchy_dir, True)

    # Create the subdirectory
    if not os.path.exists(annarchy_dir):
        os.makedirs(annarchy_dir)
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
        wfile.write(ConfigManager().get('paradigm', net_id)+', '+ANNarchy.__release__)

    sys.path.append(annarchy_dir)

def compile(
        directory='annarchy',
        clean=False,
        compiler="default",
        compiler_flags="default",
        add_sources="",
        extra_libs="",
        cuda_config={'device': 0},
        annarchy_json="",
        silent=False,
        debug_build=False,
        trace_calls=None,
        profile_enabled=False,
        net_id=0
    ):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will perform the simulation.

    The ``compiler``, ``compiler_flags`` and part of ``cuda_config`` take their default value from the configuration file ``~/.config/ANNarchy/annarchy.json``.

    The following arguments are for internal development use only:

    * **debug_build**: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    * **trace_calls**: if set to *init*, *simulate*, or *both* simulation calls inside of the C++ kernel are logged to console (default: None)
    * **profile_enabled**: creates a profilable version of ANNarchy, which logs several computation timings (default: False).

    :param directory: name of the subdirectory where the code will be generated and compiled. Default: "annarchy/".
    :param clean: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    :param compiler: C++ compiler to use. Default: g++ on GNU/Linux, clang++ on OS X. Valid compilers are [g++, clang++].
    :param compiler_flags: platform-specific flags to pass to the compiler. Defaults are defined in annarchy.json: "-march=native -O3". 
    :param cuda_config: dictionary defining the CUDA configuration for each population and projection.
    :param annarchy_json: compiler flags etc can be stored in a .json file normally placed in the home directory. With this flag one can directly assign a file location.
    :param silent: defines if status message like "Compiling... OK" should be printed.
    """
    # Check if the network has already been compiled
    if NetworkManager().get_network(net_id).compiled:
        Messages._print("compile(): the network has already been compiled, doing nothing.")
        return

    # Get command-line arguments. Note that setup() related flags has been partially parsed!
    options, unknown = ANNarchy._arg_parser.parser.parse_known_args()

    # Check for unknown flags
    if len(unknown) > 0 and ConfigManager().get('verbose', net_id):
        Messages._warning('unrecognized command-line arguments:', unknown)

    # Get CUDA configuration
    if options.gpu_device >= 0:
        cuda_config['device'] = int(options.gpu_device)

    # Check that a single backend is chosen
    if (options.num_threads != None) and (options.gpu_device >= 0):
        Messages._error('CUDA and openMP can not be active at the same time, please check your command line arguments.')

    # check if profiling was enabled by --profile
    if options.profile != None:
        profile_enabled = options.profile
        _update_global_config('profiling', options.profile)
        _update_global_config('profile_out', options.profile_out)

    # check if profiling enabled due compile()
    if profile_enabled != False and options.profile is None:
        _update_global_config('profiling', True)
    # if profiling is enabled
    if profile_enabled:
        # this will automatically create a globally available Profiler instance
        Profiler().enable_profiling()
        if ConfigManager().get('profile_out', net_id) is None:
            _update_global_config('profile_out', '.')

    # Debug the simulation kernel
    if debug_build is False:
        debug_build = options.debug  # debug build
    _update_global_config('debug', debug_build)

    # Trace function calls in simulation kernel
    if trace_calls is None:
        trace_calls = options.trace_calls  # debug build
    _update_global_config('trace_calls', trace_calls)

    # Clean
    clean = options.clean or clean # enforce rebuild

    # Compiling directory
    annarchy_dir = os.path.abspath(directory)
    if not annarchy_dir.endswith('/'):
        annarchy_dir += '/'

    # using a raw-string we can handle whitespaces in folder paths
    annarchy_dir = r'{}'.format(annarchy_dir)

    # Test if the current ANNarchy version is newer than what was used to create the subfolder
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

            elif prev_paradigm != ConfigManager().get('paradigm', net_id):
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
        add_sources=add_sources,
        extra_libs=extra_libs,
        path_to_json=annarchy_json,
        silent=silent,
        cuda_config=cuda_config,
        debug_build=debug_build,
        trace_calls=trace_calls,
        profile_enabled=profile_enabled,
        net_id=net_id
    )

    # Code Generation
    compiler.generate()

    if ConfigManager().get('verbose', net_id):
        net_str = "" if compiler.net_id == 0 else str(compiler.net_id)+" "
        Messages._print('Construct network '+net_str+'...', end=" ")

    # Create the Python objects
    _instantiate(compiler.net_id, cuda_config=compiler.cuda_config, user_config=compiler.user_config)

    # NormProjections require an update of afferent projections
    _update_num_aff_connections(compiler.net_id)

    if ConfigManager().get('verbose', net_id):
        Messages._print('OK')

    # Create a report if requested
    if options.report is not None:
        report(options.report)

def detect_cython():
    """
    Detect cython compiler and return absolute path.
    """
    # Check cython version
    with subprocess.Popen(sys.base_prefix + "/bin/cython%(major)s -V > /dev/null 2> /dev/null" % {'major': str(sys.version_info[0])}, shell=True) as test:
        if test.wait() != 0:
            cython = sys.base_prefix + "/bin/cython"
        else:
            cython = sys.base_prefix + "/bin/cython" + str(sys.version_info[0])
    # If not in the same folder as python, use the default
    with subprocess.Popen("%(cython)s -V > /dev/null 2> /dev/null" % {'cython': cython}, shell=True) as test:
        if test.wait() != 0:
            cython = shutil.which("cython"+str(sys.version_info[0]))
            if cython is None:
                cython = shutil.which("cython")
                if cython is None:
                    Messages._error("Unable to detect the path to cython.")

    return cython

def detect_cuda_arch():
    """
    For best performance, the compute compability should be mentioned to the compiler. CMake > 3.18 also enforces the
    setting of the compute-compability (see "cmake --help-policy CMP0104" for more details).
    """
    # I don't know ...
    if sys.platform.startswith('darwin'):
        return ""

    try:
        # check nvidia-smi for GPU details (only available for CUDA SDK > 11.6)
        query_result = subprocess.check_output("nvidia-smi --query-gpu=compute_cap --format=csv", shell=True)
    except:
        return ""

    # bytes to string conversion, the result contains compute_cap\nCC for each gpu\n
    query_result = query_result.decode('utf-8').split('\n')

    # NVIDIA and it's version numbering ...
    CC = int(float(query_result[1])*10)
    return """
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES {})
    endif()
""".format(CC)

class Compiler(object):
    " Main class to generate C++ code efficiently"

    def __init__(self, 
                 annarchy_dir, 
                 clean, 
                 compiler, 
                 compiler_flags, 
                 add_sources, 
                 extra_libs, 
                 path_to_json, 
                 silent, 
                 cuda_config, 
                 debug_build,
                 trace_calls,
                 profile_enabled, 
                 net_id):

        # Store arguments
        self.annarchy_dir = annarchy_dir
        self.clean = clean
        self.compiler = compiler
        self.compiler_flags = compiler_flags
        self.add_sources = add_sources
        self.extra_libs = extra_libs
        self.silent = silent
        self.cuda_config = cuda_config
        self.debug_build = debug_build
        self.trace_calls = trace_calls
        self.profile_enabled = profile_enabled
        self.net_id = net_id

        # Network to compile
        self.network = NetworkManager().get_network(net_id)

        # Aside from arguments provided to compile, some configuration is stored in annarchy.json
        if len(path_to_json) == 0:
            # check home-directory
            if os.path.exists(os.path.expanduser('~/.config/ANNarchy/annarchy.json')):
                with open(os.path.expanduser('~/.config/ANNarchy/annarchy.json'), 'r') as rfile:
                    self.user_config = json.load(rfile)
            else:
                # Set default user-defined config
                self.user_config = {
                    'openmp': {
                        'compiler': 'clang++' if sys.platform == "darwin" else 'g++',
                        'flags' : "-march=native -O3",
                    },
                    'cuda': {
                        'compiler': "nvcc",
                        'device': 0
                    }
                }

        else:
            # Load user-defined annarchy.json
            with open(path_to_json, 'r') as rfile:
                self.user_config = json.load(rfile)

        # Sanity check if the NVCC compiler is available
        if _check_paradigm("cuda", self.net_id):
            cmd = self.user_config['cuda']['compiler'] + " --version 1> /dev/null"

            if os.system(cmd) != 0:
                Messages._error("CUDA is not available on your system. Please check the CUDA installation or the annarchy.json configuration.")

            self.cuda_config['cuda_version'] = check_cuda_version(self.user_config['cuda']['compiler'])

    def generate(self):
        "Perform the code generation for the C++ code and create the Makefile."

        if Profiler().enabled or ConfigManager().get('show_time', self.net_id):
            t0 = time.time()
            if Profiler().enabled:
                Profiler().add_entry(t0, t0, "overall", "compile")

        if ConfigManager().get('verbose', self.net_id):
            net_str = "" if self.net_id == 0 else str(self.net_id)+" "
            Messages._print('Code generation '+net_str+'...', end=" ", flush=True)

        # Check that everything is allright in the structure of the network.
        check_structure(self.network.get_populations(), self.network.get_projections())

        # check if the user access some new features, or old ones which changed.
        check_experimental_features(self.network.get_populations(), self.network.get_projections())

        # Generate the code
        self.code_generation()

        # Generate the Makefile
        self.generate_makefile()

        # Copy the files if needed
        changed = self.copy_files()

        # Code generation done
        if ConfigManager().get('verbose', self.net_id):
            t1 = time.time()
            if not ConfigManager().get('show_time', self.net_id):
                Messages._print("OK", flush=True)
            else:
                Messages._print("OK (took "+str(t1-t0)+" seconds)", flush=True)

        # Shared libraries have os-dependent suffixes
        if sys.platform.startswith('linux'):
            lib_path = self.annarchy_dir + '/ANNarchyCore' + str(self.net_id) + '.so'
        elif sys.platform.startswith('darwin'):
            lib_path = self.annarchy_dir + '/ANNarchyCore' + str(self.net_id) + '.dylib'
        else:
            raise NotImplementedError

        # Perform compilation if something has changed
        if changed or not os.path.isfile(lib_path):
            self.compilation()

        # Set the compilation directory in the networks
        if ConfigManager().get('debug', self.net_id) or ConfigManager().get('disable_shared_library_time_offset', self.net_id):
            # In case of debugging or high-throughput simulations we want to
            # disable the trick below
            self.network.directory = self.annarchy_dir
        else:
            # Store the library in random subfolder
            # We circumvent with this an issue with reloading of shared libraries
            # see PEP 489: (https://www.python.org/dev/peps/pep-0489/) for more details
            directory = self.annarchy_dir+'/run_'+str(time.time())
            self.network.directory = directory
            os.mkdir(directory)
            shutil.copy(lib_path, directory)

        # Tell the networks they have been compiled
        self.network.compiled = True
        if Profiler().enabled:
            t1 = time.time()
            Profiler().update_entry(t0, t1, "overall", "compile")

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
                    not file == "codegen.log" and \
                    not filecmp.cmp(self.annarchy_dir+'/generate/net' + str(self.net_id) + '/' + file,
                                    self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' + file):


                    shutil.copy(self.annarchy_dir+'/generate//net'+ str(self.net_id) + '/' + file, # src
                                self.annarchy_dir+'/build/net'+ str(self.net_id) + '/' +file # dest
                               )
                    changed = True

                    if ConfigManager().get('verbose', self.net_id):
                        print(file, 'has changed')
                        # For debugging
                        # with open(self.annarchy_dir+'/generate/net'+ str(self.net_id) + '/' + file, 'r') as rfile:
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
            if ConfigManager().get('verbose', self.net_id):
                msg = 'Compiling with ' + self.compiler + ' ' + self.compiler_flags
            else:
                msg = 'Compiling '
            if self.net_id > 0:
                msg += 'network ' + str(self.net_id)
            msg += '...'
            Messages._print(msg, end=" ", flush=True)
            if ConfigManager().get('show_time', self.net_id) or Profiler().enabled:
                t0 = time.time()

        target_dir = self.annarchy_dir + '/build/net'+ str(self.net_id)

        # using a raw-string we can handle whitespaces in folder paths
        target_dir = r'{}'.format(target_dir)

        # Switch to the build directory
        cwd = os.getcwd()
        os.chdir(target_dir)
        
        # CMake is quite talky by default (printing out compiler versions etc.)
        # We reduce the number of printed messages except the user enabled verbose mode.
        verbose = "> compile_stdout.log 2> compile_stderr.log" if not ConfigManager().get('verbose', self.net_id) else ""

        # Generate the Makefile from CMakeLists
        make_process = subprocess.Popen("cmake -S \"{}\" -B \"{}\" {}".format(target_dir, target_dir, verbose), shell=True)
        if make_process.wait() != 0:
            Messages._error('CMake generation failed.')


        # Start the compilation
        verbose = "> compile_stdout.log 2> compile_stderr.log" if not ConfigManager().get('verbose', self.net_id) else ""
        make_process = subprocess.Popen("make -j4" + verbose, shell=True)

        # Check for errors
        if make_process.wait() != 0:
            with open('compile_stderr.log', 'r') as rfile:
                msg = rfile.read()
            with open(self.annarchy_dir + '/compilation', 'w') as wfile:
                wfile.write("0")
            Messages._print(msg)
            try:
                if sys.platform.startswith('linux'):
                    os.remove('ANNarchyCore'+str(self.net_id)+'.so')
                elif sys.platform.startswith('darwin'):
                    os.remove('ANNarchyCore'+str(self.net_id)+'.dylib')
                else:
                    raise NotImplementedError
            except:
                pass
            Messages._error('Compilation failed.')
            
        else: # Note that the last compilation was successful
            with open(self.annarchy_dir + '/compilation', 'w') as wfile:
                wfile.write("1")

        # Return to the current directory
        os.chdir(cwd)

        if not self.silent:
            t1 = time.time()

            if not ConfigManager().get('show_time', self.net_id):
                Messages._print('OK', flush=True)
            else:
                Messages._print('OK (took '+str(t1 - t0)+'seconds.', flush=True)

            if Profiler().enabled:
                Profiler().add_entry(t0, t1, "compilation", "compile")

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
            cpu_flags = "-O0 -g -D_DEBUG -march=native"

        # trace of function calls should be possible without debug mode
        if self.trace_calls is not None:
            if self.trace_calls in ["init", "both"]:
                cpu_flags += " -D_TRACE_INIT"

            if self.trace_calls in ["simulate", "both"]:
                cpu_flags += " -D_TRACE_SIMULATION_STEPS"

        if self.profile_enabled:
            cpu_flags += " -g"

        # OpenMP flag
        omp_flag = ""
        if ConfigManager().get('paradigm', self.net_id) == "openmp" :
            omp_flag = "-fopenmp"

        # Disable openMP parallel RNG?
        if ConfigManager().get('disable_parallel_rng', self.net_id) and _check_paradigm("openmp", self.net_id):
            cpu_flags += " -D_DISABLE_PARALLEL_RNG "

        # Disable auto-vectorization
        if ConfigManager().get('disable_SIMD_Eq', self.net_id) and _check_paradigm("openmp", self.net_id):
            cpu_flags += " -fno-tree-vectorize"

        # Cuda Library and Compiler
        #
        # hdin (22.03.2016): we should verify in the future, if compute_35 remains as best
        # configuration for Keplar and upwards.
        cuda_gen = ""
        gpu_flags = ""
        gpu_compiler = "nvcc"
        gpu_ldpath = ""
        xcompiler_flags = ""
        if sys.platform.startswith('linux') and ConfigManager().get('paradigm', self.net_id) == "cuda":
            cuda_gen = "" # TODO: -arch sm_%(ver)s

            if self.debug_build:
                gpu_flags = "-g -G -D_DEBUG"

            # read the config file for the cuda lib path
            if 'cuda' in self.user_config.keys():
                gpu_compiler = self.user_config['cuda']['compiler']
                gpu_ldpath = '-L' + self.user_config['cuda']['path'] + '/lib'
                gpu_flags += self.user_config['cuda']['flags']

            # -Xcompiler expects the arguments seperated by ','
            if len(cpu_flags.strip()) > 0:
                xcompiler_flags = cpu_flags.replace(" ",",")

        # Extra libs from extensions such as opencv
        libs = self.extra_libs
        for lib in extra_libs:
            libs += str(lib) + ' '


        if ConfigManager().get('paradigm', self.net_id) == "cuda":
            set_cuda_arch = detect_cuda_arch()
        else:
            set_cuda_arch = ""

        # Include path to Numpy is not standard on all distributions
        numpy_include = np.get_include()

        # ANNarchy default header: sparse matrix formats
        annarchy_include = ANNarchy.__path__[0]+'/include'

        # Thirdparty includes (C++ files)
        thirdparty_include = ANNarchy.__path__[0]+'/thirdparty'

        # The connector module needs to reload some header files,
        # ANNarchy.__path__ provides the installation directory
        path_to_cython_ext = '-I'+ANNarchy.__path__[0]+'/core/cython_ext/\" \"-I'+ANNarchy.__path__[0][:-8]

        # Create Makefiles depending on the target platform and parallel framework
        if sys.platform.startswith('linux'): # Linux systems
            if ConfigManager().get('paradigm', self.net_id) == "cuda":
                makefile_template = linux_cuda_template
            else:
                makefile_template = linux_omp_template

        elif sys.platform == "darwin":   # mac os
            if self.compiler == 'clang++':
                makefile_template = osx_clang_template
                if ConfigManager().get('num_threads', self.net_id) == 1: # clang should report that it does not support openmp
                    omp_flag = ""
            else:
                makefile_template = osx_gcc_template

        else: 
            # Windows: to test....
            Messages._warning("Compilation on windows is not supported yet. We recommend to use WSL on windows systems.")

        # Gather all Makefile flags
        makefile_flags = {
            'compiler': self.compiler,
            'add_sources': self.add_sources,
            'cpu_flags': cpu_flags,
            'cuda_gen': cuda_gen,
            'gpu_compiler': gpu_compiler,
            'gpu_flags': gpu_flags,
            'xcompiler_flags': xcompiler_flags,
            'gpu_ldpath': gpu_ldpath,
            'openmp': omp_flag,
            'set_cuda_arch': set_cuda_arch,
            'extra_libs': libs,
            'numpy_include': numpy_include,
            'annarchy_include': annarchy_include,
            'thirdparty_include': thirdparty_include,
            'net_id': self.net_id,
            'cython_ext': path_to_cython_ext
        }

        # Write the Makefile to the disk
        with open(self.annarchy_dir + '/generate/net'+ str(self.net_id) + '/CMakeLists.txt', 'w') as wfile:
            wfile.write(makefile_template % makefile_flags)


    def code_generation(self):
        """
        Code generation dependent on paradigm
        """
        # First, we remove previously created files
        target_folder = self.annarchy_dir+'/generate/net'+ str(self.net_id)
        if ConfigManager().get("verbose", self.net_id):
            print('\n\nCheck target folder:', target_folder)
        for file in os.listdir(target_folder):
            if ConfigManager().get("verbose", self.net_id):
                print("  - remove file:", file)
            os.remove(target_folder+'/'+file)

        # Then, we generate the code for the current network
        generator = CodeGenerator(
            self.annarchy_dir, 
            self.net_id, 
            self.cuda_config)
        
        generator.generate()


def load_cython_lib(libname, libpath):
    """
    Load the shared library created by Cython using importlib. Follows the example
    "Multiple modules in one library" in PEP 489.

    TODO:

    As described in PEP 489 "Module Reloading" a reloading of dynamic extension modules is
    not supported. This leads to some problems for our reusage of the ANNarchyCore library ...

    Sources:

    PEP 489: (https://www.python.org/dev/peps/pep-0489/)
    """
    # create a loader to mimic find module
    loader = importlib.machinery.ExtensionFileLoader(libname, libpath)
    spec = importlib.util.spec_from_loader(libname, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module

def _instantiate(net_id, import_id=-1, cuda_config=None, user_config=None, core_list=None):
    """
    After every is compiled, actually create the Cython objects and bind them to the Python ones.
    """
    if Profiler().enabled:
        t0 = time.time()
        Profiler().add_entry(t0, t0, "overall", "instantiate") # placeholder, to have the correct ordering

    # parallel_run(number=x) defines multiple networks (net_id) but only network0 is compiled
    if import_id < 0:
        import_id = net_id

    # subdirectory where the library lies
    annarchy_dir = NetworkManager().get_network(net_id=import_id).directory
    libname = 'ANNarchyCore' + str(import_id)
    if sys.platform.startswith('linux'):
        libpath = annarchy_dir + '/' + libname + '.so'
    elif sys.platform.startswith('darwin'):
        libpath = annarchy_dir + '/' + libname + '.dylib'
    else:
        raise NotImplementedError

    # Load the library
    cython_module = load_cython_lib(libname, libpath)
    NetworkManager().get_network(net_id=net_id).instance = cython_module

    # Sets the desired number of threads and execute thread placement.
    # This must be done before any other objects are initialized.
    if _check_paradigm("openmp", net_id):
        # check for global setting
        if core_list is None:
            core_list = ConfigManager().get('visible_cores', net_id)

        # the user configured a setup
        if core_list != []:
            # some sanity check
            if len(core_list) > multiprocessing.cpu_count():
                Messages._error("The length of core ids provided to setup() is larger than available number of cores")

            if len(core_list) < ConfigManager().get('num_threads', net_id):
                Messages._error("The list of visible cores should be at least the number of cores.")

            if np.amax(np.array(core_list)) > multiprocessing.cpu_count():
                Messages._error("At least one of the core ids provided to setup() is larger than available number of cores")

            if len(core_list) != len(list(set(core_list))):
                Messages._warning("The provided core list contains doubled entries - is this intended?")

            cython_module.set_number_threads(ConfigManager().get('num_threads', net_id), np.array(core_list))

        else:
            # HD (26th Oct 2020): the current version of psutil only consider one CPU socket
            #                     but there is a discussion of adding multi-sockets, so we could
            #                     re-add this code later ...
            """
            num_cores = psutil.cpu_count(logical=False)
            # Check if the number of threads make sense
            if num_cores < ConfigManager().get('num_threads', net_id):
                Messages._warning("The number of threads =", ConfigManager().get('num_threads', net_id), "exceeds the number of available physical cores =", num_cores)

            # ANNarchy should run only on physical cpu cores
            core_list = np.arange(0, num_cores)
            """
            cython_module.set_number_threads(ConfigManager().get('num_threads', net_id), [])

        if ConfigManager().get('num_threads', net_id) > 1:
            if ConfigManager().get('verbose', net_id):
                Messages._print('Running simulation with', ConfigManager().get('num_threads', net_id), 'threads.')
        else:
            if ConfigManager().get('verbose', net_id):
                Messages._print('Running simulation single-threaded.')

    elif _check_paradigm("cuda", net_id):
        # check if there is a configuration,
        # otherwise fall back to default device
        device = 0
        if user_config is not None and 'cuda' in user_config.keys():
            device = int(cuda_config['device'])
        elif cuda_config is not None:
            device = int(user_config['cuda']['device'])

        if ConfigManager().get('verbose', net_id):
            Messages._print('Setting GPU device', device)

        # Set the CUDA device
        cython_module.set_device(device)

    else:
        raise NotImplementedError

    # Instantiate CPP objects
    cython_module.pyx_create()

    # Configure seeds for C++ random number generators
    # Required for state updates and also (in future) construction of connectivity
    seed = ConfigManager().get('seed', net_id)
    if seed is None:
        seed = int(time.time())

    if not ConfigManager().get('disable_parallel_rng', net_id):
        cython_module.set_seed(seed, ConfigManager().get('num_threads', net_id), ConfigManager().get('use_seed_seq', net_id))
    else:
        cython_module.set_seed(seed, 1, ConfigManager().get('use_seed_seq', net_id))

    if Profiler().enabled:
        # register the CPP profiling instance
        # Attention: since ANNarchy 5.0 this need to be instantiated before any other cpp object.
        Profiler()._cpp_profiler = NetworkManager().get_network(net_id).instance.Profiling_wrapper()

    # Bind the py extensions to the corresponding python objects
    for pop in NetworkManager().get_network(net_id).get_populations():
        if ConfigManager().get('verbose', net_id):
            Messages._print('Instantiate population ( name =', pop.name, ', size =', pop.size,')')
        if ConfigManager().get('show_time', net_id):
            t0 = time.time()

        # Instantiate the population
        pop._instantiate(cython_module)

        if ConfigManager().get('show_time', net_id):
            Messages._print('  instantiate of the popukatuib took', (time.time()-t0)*1000, 'milliseconds')

    # Instantiate projections
    for proj in NetworkManager().get_network(net_id).get_projections():
        if ConfigManager().get('verbose', net_id):
            Messages._print('Instantiate projection ( pre =', proj.pre.name, ', post =', proj.post.name, ', target =', proj.target, ')')
        if ConfigManager().get('show_time', net_id):
            t0 = time.time()

        # Create the projection
        proj._instantiate(cython_module)

        if ConfigManager().get('show_time', net_id):
            Messages._print('  instantiate of the projection took', (time.time()-t0)*1000, 'milliseconds')

    # Finish to initialize the network
    cython_module.pyx_initialize(ConfigManager().get('dt', net_id))

    # Set the user-defined constants
    for obj in NetworkManager().get_network(net_id).get_constants():
        getattr(cython_module, 'set_'+obj.name)(obj.value)

    # Transfer initial values
    for pop in NetworkManager().get_network(net_id).get_populations():
        if ConfigManager().get('verbose', net_id):
            Messages._print('Initializing C++ counterpart of population', pop.name)
        pop._init_attributes()
    for proj in NetworkManager().get_network(net_id).get_projections():
        if ConfigManager().get('verbose', net_id):
            Messages._print('Initializing C++ counterpart of projection', proj.name)
        proj._init_attributes()

    # Start the monitors
    for monitor in NetworkManager().get_network(net_id).get_monitors():
        monitor._init_monitoring()

    if Profiler().enabled:
        t1 = time.time()
        Profiler().update_entry(t0, t1, "overall", "instantiate")
