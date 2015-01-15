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
import os, sys
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
 
def _folder_management(profile_enabled, clean):
    """
    ANNarchy is provided as a python package. For compilation a local folder
    'annarchy' is created in the current working directory.
    
    *Parameter*:
    
    * *profile_enabled*: copy needed data for profile extension
    """
        
    # Verbose
    if Global.config['verbose']:
        Global._print("Create 'annarchy' subdirectory.")

    if clean or profile_enabled:
        shutil.rmtree(Global.annarchy_dir, True)

    if not os.path.exists(Global.annarchy_dir):    
        os.mkdir(Global.annarchy_dir)
        os.mkdir(Global.annarchy_dir+'/build')
        
    # Create the generate subfolder
    shutil.rmtree(Global.annarchy_dir+'/generate', True)
    os.mkdir(Global.annarchy_dir+'/generate')

    # Save current ANNarchy version
    with open(Global.annarchy_dir+'/release', 'w') as f:
        f.write(ANNarchy.__release__)

    sys.path.append(Global.annarchy_dir)

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

def compile(clean=False, populations=None, projections=None, cpp_stand_alone=False, debug_build=False, profile_enabled = False):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will perform the simulation.
    
    *Parameters*:

    * **clean**: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * **populations**: list of populations which should be compiled. If set to None, all available populations will be used.
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
    elif Global.config['num_threads'] == None:
        Global.config['num_threads'] = 1
    if options.verbose != None:
        Global.config['verbose'] = options.verbose
    if options.profile != None:
        profile_enabled = options.profile
        Global.config['profiling']= options.profile

    debug_build = options.debug # debug build
    clean = options.clean # enforce rebuild

    if populations == None: # Default network
        populations = Global._populations

    if projections == None: # Default network
        projections = Global._projections

    # Test if the current ANNarchy version is newer than what was used to create the subfolder
    from pkg_resources import parse_version
    if os.path.isfile(Global.annarchy_dir+'/release'):
        with open(Global.annarchy_dir+'/release', 'r') as f:
            prev_release = f.read().strip()
            if parse_version(prev_release) < parse_version(ANNarchy.__release__):
                print 'ANNarchy has been updated, recompiling...'
                clean = True
    else:
        clean = True

    # Manage the compilation subfolder
    _folder_management(profile_enabled, clean)
    
    # Create a Generator object
    generator = Generator(clean, cpp_stand_alone, debug_build, profile_enabled, 
                 populations, projections)
    generator.generate()
    
class Generator(object):
    " Main class to generate C++ code efficiently"
      
    def __init__(self, clean, cpp_stand_alone, debug_build, profile_enabled, 
                 populations, projections): 
        
        # Store arguments
        self.clean = clean
        self.cpp_stand_alone = cpp_stand_alone
        self.debug_build = debug_build
        self.profile_enabled = profile_enabled
        self.populations = populations
        self.projections = projections
        
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
        if changed or not os.path.isfile(Global.annarchy_dir+'/ANNarchyCore.so'):
            self.compilation()

            # Return to the current directory
            os.chdir('..')
                
        Global.set_compiled()
        
        # Create the Python objects                
        self.instantiate()    

    def copy_files(self, clean):
        " Copy the generated files in the build/ folder if needed."
        changed = False
        if clean:
            for file in os.listdir(Global.annarchy_dir+'/generate'):
                shutil.copy(Global.annarchy_dir+'/generate/'+file, # src
                            Global.annarchy_dir+'/build/'+file # dest
                )
            changed = True
        else: # only the ones which have changed
            import filecmp
            for file in os.listdir(Global.annarchy_dir+'/generate'):
                if  not os.path.isfile(Global.annarchy_dir+'/build/'+file) or \
                    not filecmp.cmp( Global.annarchy_dir+'/generate/'+file, 
                                    Global.annarchy_dir+'/build/'+file) :
                    shutil.copy(Global.annarchy_dir+'/generate/'+file, # src
                                Global.annarchy_dir+'/build/'+file # dest
                    )
                    changed = True
        return changed

    def compilation(self):
        """ Create ANNarchyCore.so and py extensions if something has changed."""

        # set OpenMP optimization flags
        #os.environ["OMP_PROC_BIND"] = "true"
 
        Global._print('Compiling ... ')
        if Global.config['show_time']:
            t0 = time.time()

        os.chdir(Global.annarchy_dir)
        if sys.platform.startswith('linux'): # Linux systems

            if Global.config['paradigm'] != "openmp":
                from .CudaCheck import CudaCheck as CC
                a = CC()

            if not self.debug_build:
                cpu_flags = "-march=native -O2"
                gpu_flags = ""
            else:
                cpu_flags = "-O0 -g -D_DEBUG"
                gpu_flags = ""
                
            if self.profile_enabled:
                cpu_flags += " -g"
                gpu_flags += " -G -lineinfo"
                extra_libs.append("-lpapi")
                
            libs = ""
            for l in extra_libs:
                libs += str(l) + ' '
    
            py_version = "%(major)s.%(minor)s" % { 'major': sys.version_info[0],
                                                   'minor': sys.version_info[1] }

            numpy_include = np.get_include()
    
            # generate Makefile
            src = """# Makefile generated by ANNarchy
all:
\t@echo "Use either openmp, cuda_20 or cuda_35"
    
openmp:
\tcython build/ANNarchyCore.pyx --cplus
\tg++ -fPIC -shared %(cpu_flags)s -fpermissive -std=c++0x -I. -I/usr/include/python%(py_version)s -I %(numpy_include)s -fopenmp build/*.cpp %(libs)s -o ANNarchyCore.so

# CC 2.0 (Tesla)
cuda_20:
\tcython build/ANNarchyCore.pyx --cplus
\tnvcc -gencode arch=compute_20,code=compute_20 %(gpu_flags)s -c build/cuANNarchy.cu -Xcompiler -fPIC -o build/cuANNarchy.o
\tg++ -fPIC -shared %(cpu_flags)s -fpermissive -std=c++0x -I. -I/usr/include/python2.7 -fopenmp build/*.cpp build/cuANNarchy.o -lcudart -lcurand %(libs)s -o ANNarchyCore.so

# CC 3.5 (Keplar)
cuda_35:
\tcython build/ANNarchyCore.pyx --cplus
\tnvcc -gencode arch=compute_35,code=compute_35 %(gpu_flags)s -c build/cuANNarchy.cu -Xcompiler -fPIC -o build/cuANNarchy.o
\tg++ -fPIC -shared %(cpu_flags)s -fpermissive -std=c++0x -I. -I/usr/include/python2.7 -fopenmp build/*.cpp build/cuANNarchy.o -lcudart -lcurand %(libs)s -o ANNarchyCore.so

clean:
\trm -rf build/*.o
\trm -rf build/*.so
""" % {'cpu_flags': cpu_flags, 'gpu_flags': gpu_flags, 'libs': libs, 'py_version': py_version, 'numpy_include': numpy_include}

            # Write the Makefile to the disk
            with open('Makefile', 'w') as wfile:
                wfile.write(src)
                
            # Start the compilation
            try:
                if Global.config['paradigm'] == "cuda":
                    from .CudaCheck import CudaCheck
                    cu_version = CudaCheck().version()

                    if cu_version >= (3,0):
                        print "Build with cuda 3.x"
                        subprocess.check_output("make cuda_35 -j4 ", 
                                                 shell=True)
                    else:
                        print "Build with cuda 2.x"
                        subprocess.check_output("make cuda_20 -j4 ", 
                                                 shell=True)                    
                elif Global.config['paradigm'] == "openmp":
                    subprocess.check_output("make openmp -j4 ", 
                                            shell=True)
                else:
                    Global._warning("unknown paradigm '"+Global.config['paradigm']+"': set default openmp")
                    subprocess.check_output("make openmp -j4 ", 
                                            shell=True)
                    
                subprocess.check_output("make -j4 > compile_stdout.log 2> compile_stderr.log", 
                                        shell=True)

            except subprocess.CalledProcessError:
                with open('compile_stderr.log', 'r') as rfile:
                    msg = rfile.read()
                Global._print(msg)
                Global._error('Compilation failed.')
                try:
                    os.remove('ANNarchyCore.so')
                except:
                    pass
                exit(0)
    
        else: # Windows: to test....
            pass

        Global._print('OK')
        if Global.config['show_time']:
            Global._print('Compilation took', time.time() - t0, 'seconds.')

                
    def instantiate(self):
        """ After every is compiled, actually create the Cython objects and 
            bind them to the Python ones."""

        if Global.config['verbose']:
            Global._print('Building network ...')

        # Import the Cython library
        cython_module = __import__('ANNarchyCore')
        global _network
        Global._network = cython_module

        # Bind the py extensions to the corresponding python objects
        for pop in self.populations:
            if Global.config['verbose']:
                Global._print('Creating population', pop.name)
            if Global.config['show_time']:
                t0 = time.time()
            
            # Instantiate the population
            pop._instantiate(cython_module)

            if Global.config['show_time']:
                Global._print('Creating', pop.name, 'took', (time.time()-t0)*1000, 'milliseconds') 
                            
        # Instantiate projections
        for proj in self.projections:
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
        for pop in self.populations:
            if Global.config['verbose']:
                Global._print('Initializing population', pop.name)
            pop._init_attributes()
        for proj in self.projections:
            if Global.config['verbose']:
                Global._print('Initializing projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')  
            proj._init_attributes()

        # Sets the desired number of threads
        cython_module.set_number_threads(Global.config['num_threads'])
            
            
    def code_generation(self, cpp_stand_alone, profile_enabled, clean):
        """
        Code generation.
        """
        if Global.config['verbose']:
            print('\nGenerate code ...')
        
        if Global.config['paradigm'] == "openmp":
            # Select paradigm here....
            from .OMP.OMPGenerator import OMPGenerator
            generator = OMPGenerator(self.populations, self.projections)
            generator.generate()
        else:
            from .CUDA.CUDAGenerator import CUDAGenerator
            generator = CUDAGenerator(self.populations, self.projections)
            generator.generate()

        if self.profile_enabled:
            from .Profile.ProfileGenerator import ProfileGenerator
            prof_generator = ProfileGenerator(self.populations, self.projections)
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
