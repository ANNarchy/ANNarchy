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

# ANNarchy core informations
import ANNarchy
import ANNarchy.core.Global as Global
from ANNarchy.parser.Analyser import Analyser

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
   
    # first argument is python script itself.
    for arg in sys.argv[1:]:
        if arg == '--clean':
            clean = True
        elif arg == '--debug':
            debug_build = True
        elif arg == '--verbose':
            Global.config['verbose'] = True
        elif arg == '--profile':
            profile_enabled = True
        elif str(arg).find('-j')!= -1:
            try:
                num_threads = int(arg.replace('-j',''))
                Global.config['num_threads'] = num_threads
                Global._debug( 'use', num_threads, 'threads')
            except:
                Global._error( 'wrong format, expected -jx')
        else:
            Global._error( 'unknown command line argument', arg )
    
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

        # Generate the code
        self.code_generation(self.cpp_stand_alone, self.profile_enabled, 
                                                        self.clean)
        # Copy the files if needed
        changed = self.copy_files(self.clean)
        
        # Perform compilation if something has changed
        if changed:
            self.compilation()
                
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
 
        Global._print('Compiling ... ')
        if Global.config['show_time']:
            t0 = time.time()
                                
        os.chdir(Global.annarchy_dir)
        if sys.platform.startswith('linux'): # Linux systems
            if not self.debug_build:
                flags = "-O2 "
            else:
                flags = "-O0 -g -D_DEBUG"
                
            if self.profile_enabled:
                flags += "-g "
                
            libs = ""
            for l in extra_libs:
                libs += str(l) + ' '
    
            py_version = "%(major)s.%(minor)s" % { 'major': sys.version_info[0],
                                                   'minor': sys.version_info[1] }
    
            # generate Makefile
            src = """# Makefile generated by ANNarchy
all:
\t@echo "Use either openmp, cuda_20 or cuda_35"
    
openmp:
\tcython build/ANNarchyCore.pyx --cplus
\tg++ -march=native -O2 -shared -fPIC -fpermissive -std=c++0x -I. -I/usr/include/python2.7 -fopenmp build/*.cpp -o ANNarchyCore.so

# CC 2.0 (Tesla)
cuda_20:
\tcython build/ANNarchyCore.pyx --cplus
\tnvcc -gencode arch=compute_20,code=compute_20 -c build/cuANNarchy.cu -o build/cuANNarchy.o
\tg++ -march=native -O2 -shared -fPIC -fpermissive -std=c++0x -I. -I/usr/include/python2.7 -fopenmp build/*.cpp -lcudart -o ANNarchyCore.so

# CC 3.5 (Keplar)
cuda_35:
\tcython build/ANNarchyCore.pyx --cplus
\tnvcc -g -lineinfo -gencode arch=compute_35,code=compute_35 -c build/cuANNarchy.cu -Xcompiler -fPIC -o build/cuANNarchy.o
\tg++ -fPIC -g -march=native -O2 -shared -fPIC -fpermissive -std=c++0x -I. -I/usr/include/python2.7 -fopenmp build/*.cpp build/cuANNarchy.o -lcudart -o ANNarchyCore.so

clean:
\trm -rf build/*.o
\trm -rf build/*.so
"""

            # Write the Makefile to the disk
            with open('Makefile', 'w') as wfile:
                wfile.write(src)
            
            if Global.config['verbose'] == False:
                pipe_to_cmd_line = "> compile_stdout.log 2> compile_stderr.log"
            else:
                pipe_to_cmd_line = ""
                
            # Start the compilation
            try:
                if Global.config['paradigm'] == "cuda_35":
                    subprocess.check_output("make cuda_35 -j4 "+ pipe_to_cmd_line, 
                                            shell=True)
                elif Global.config['paradigm'] == "cuda_20":
                    subprocess.check_output("make cuda_20 -j4 "+ pipe_to_cmd_line, 
                                            shell=True)                    
                elif Global.config['paradigm'] == "openmp":
                    subprocess.check_output("make openmp -j4 "+ pipe_to_cmd_line, 
                                            shell=True)
                else:
                    Global._warning("unknown paradigm '"+Global.config['paradigm']+"': set default openmp")
                    subprocess.check_output("make openmp -j4 "+ pipe_to_cmd_line, 
                                            shell=True)
                    
            except subprocess.CalledProcessError:
                Global._error('Compilation failed.\nCheck the compilation logs in annarchy/compile_stderr.log')
                exit(0)
    
        else: # Windows: to test....
            pass
        
        Global._compiled = True

        Global._print('OK')
        if Global.config['show_time']:
            Global._print('Compilation took', time.time() - t0, 'seconds.')
                
    def instantiate(self):
        """ After every is compiled, actually create the Cython objects and 
            bind them to the Python ones."""
        
        if Global.config['verbose']:
            Global._print('Building network ...')

        # Return to the current directory
        os.chdir('..')

        # Import the Cython library
        cython_module = __import__('ANNarchyCore')
        global _network
        Global._network = cython_module

        # Bind the py extensions to the corresponding python objects
        for name, pop in self.populations.iteritems():
            if Global.config['verbose']:
                Global._print('Create population', pop.name)
            if Global.config['show_time']:
                t0 = time.time()
            
            # Instantiate the population
            pop._instantiate(cython_module)

            if Global.config['show_time']:
                Global._print('Creating', pop.name, 'took', (time.time()-t0)*1000, 'milliseconds') 
                            
        # Instantiate projections
        for name, proj in self.projections.iteritems():
            if Global.config['verbose']:
                Global._print('Creating projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')        
            if Global.config['show_time']:
                t0 = time.time()
            
            # Create the projection
            proj._instantiate(cython_module)
            
            if Global.config['show_time']:
                Global._print('    took', (time.time()-t0)*1000, 'milliseconds')
    

        # Finish to initialize the network, especially the rng
        # Must be called after the pops and projs are created!
        cython_module.pyx_create(Global.config['dt'])

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