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

# ANNarchy core informations
import ANNarchy4
import ANNarchy4.core.Global as Global
from ANNarchy4.parser.Analyser import Analyser, _extract_functions
from ANNarchy4.generator.PopulationGenerator import RatePopulationGenerator, SpikePopulationGenerator  
from ANNarchy4.generator.ProjectionGenerator import RateProjectionGenerator, SpikeProjectionGenerator  

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
        
    # Directory where the source files reside
    sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')

    if clean or profile_enabled:
        shutil.rmtree(Global.annarchy_dir, True)

    if not os.path.exists(Global.annarchy_dir):    
        os.mkdir(Global.annarchy_dir)
        os.mkdir(Global.annarchy_dir+'/pyx')
        os.mkdir(Global.annarchy_dir+'/build')
        
    # Create the generate subfolder
    shutil.rmtree(Global.annarchy_dir+'/generate', True)
    os.mkdir(Global.annarchy_dir+'/generate')
    os.mkdir(Global.annarchy_dir+'/generate/pyx')
    os.mkdir(Global.annarchy_dir+'/generate/build')

    # cpp / h files
    for cfile in os.listdir(sources_dir+'/cpp'):
        shutil.copy(sources_dir+'/cpp/'+cfile, # src
                    Global.annarchy_dir+'/generate/build/'+cfile # dest
                    )
    # pyx files
    for pfile in os.listdir(sources_dir+'/pyx'):
        shutil.copy(sources_dir+'/pyx/'+pfile, #src
                    Global.annarchy_dir+'/generate/pyx/'+pfile #dest
                    )
    # profile files
    if profile_enabled:
        profile_sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../extensions/Profile')    
        shutil.copy(profile_sources_dir+'/Profile.cpp', Global.annarchy_dir+'/generate/build')
        shutil.copy(profile_sources_dir+'/Profile.h', Global.annarchy_dir+'/generate/build')
        shutil.copy(profile_sources_dir+'/cy_profile.pyx', Global.annarchy_dir+'/generate/pyx')

    sys.path.append(Global.annarchy_dir)

def _update_float_prec(filename):
    """
    all code generated files will be updated
    """
    code = ''
    with open(filename, mode = 'r') as r_file:
        for a_line in r_file:
            if Global.config['float_prec']=='single':
                code += a_line.replace("DATA_TYPE","float")
            else:
                code += a_line.replace("float","double").replace("DATA_TYPE","double")

    with open(filename, mode='w') as w_file:
        w_file.write(code)

            
    
def compile(clean=False, populations=None, projections=None, cpp_stand_alone=False, debug_build=False, profile_enabled = False):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will carry the simulation.
    
    *Parameters*:

    * *clean*: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * *populations*: set of populations which should be compiled. If set to None, all available populations will be used.
    * *projections*: set of populations which should be compiled. If set to None, all available populations will be used.
    * *cpp_stand_alone*: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
    * *debug_build*: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    * *profile_enabled*: creates a profilable version of ANNarchy, which logs several computation timings (default: False).
    """
   
    # first argument is python script itself.
    for arg in sys.argv[1:]:
        if arg == '--clean':
            clean = True
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
            
    # Test if profiling is enabled
    if profile_enabled:
        try:
            from ANNarchy4.extensions import Profile
        except ImportError:
            Global._error( 'Profile extension was not found.' )
            profile_enabled = False

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
        # Call the analyser to parse all equations
        self.analyser = Analyser(self.populations, self.projections)
        success = self.analyser.analyse()
        if not success:
            Global._error('The network can not be generated.')
            exit(0)
        
        # Generate the code
        self.code_generation(self.cpp_stand_alone, self.profile_enabled, 
                                                        self.clean)
        
        # Test if the code has changed last compilation
        changed_cpp, changed_pyx = self.test_changed()
        
        # Perform compilation if something has changed
        self.partial_compilation(changed_cpp, changed_pyx)
                
        # Create the Python objects    
        if not self.cpp_stand_alone:        
            if Global.config['verbose']:
                Global._print('Building network ...')
            self.instantiate()
    
        else:
            #abort the application after compiling ANNarchyCPP
            Global._print('\nCompilation process of ANNarchyCPP completed successful.\n')
            exit(0)
            
    def code_generation(self, cpp_stand_alone, profile_enabled, clean):
        """
        Code generation for each population respectively projection object the user defined. 
        
        After this the ANNarchy main header is expanded by the corresponding headers.
        """
        if Global.config['verbose']:
            print('\nGenerate code ...')
            
        # Generate code for the analysed pops and projs    
        
        # create population cpp class for each neuron
        for name, desc in self.analyser.analysed_populations.iteritems():
            if desc['type'] == 'rate':
                pop_generator = RatePopulationGenerator(name, desc)
            elif desc['type'] == 'spike':
                pop_generator = SpikePopulationGenerator(name, desc)
            pop_generator.generate(Global.config['verbose'])
        
        # create projections cpp class for each synapse
        for name, desc in self.analyser.analysed_projections.iteritems():
            if desc['type'] == 'rate':
                proj_generator = RateProjectionGenerator(name, desc)
            elif desc['type'] == 'spike':
                proj_generator = SpikeProjectionGenerator(name, desc)
            proj_generator.generate(Global.config['verbose'])
    
        # Create default files mainly based on the number of populations and projections
        if Global.config['verbose']:
            print('\nCreate Includes.h ...')
        self.create_includes()
    
        if Global.config['verbose']:
            print('\nUpdate ANNarchy.h ...')
        self.update_annarchy_header()
     
        if Global.config['verbose']:
            print('\nUpdate global header ...')
        self.update_global_header()
     
        if Global.config['verbose']:
            print('\nGenerate py extensions ...')
        self.generate_py_extension()
         
        # Change the float precision in every file
        os.chdir(Global.annarchy_dir+'/generate/build')
        cpp_src = filter(os.path.isfile, os.listdir('.'))
        for file in cpp_src:
            _update_float_prec(file)
                 
        os.chdir(Global.annarchy_dir+'/generate/pyx')
        pyx_src = filter(os.path.isfile, os.listdir('.'))
        for file in pyx_src:
            _update_float_prec(file)
        os.chdir(Global.annarchy_dir)  
        
    def test_changed(self):  
        " Test if the code generation has changed since last time."
        def copy_changed(folder):
            changed = False
            # Copy the files which have changed 
            for file in os.listdir(Global.annarchy_dir+'/generate/' + folder):
                # If the file does not exist in build/ or pyx/, 
                # or if the newly generated file is different, copy the new file
                if not os.path.isfile(Global.annarchy_dir+'/'+folder+'/'+file) or \
                   not filecmp.cmp(Global.annarchy_dir+'/generate/'+folder+'/'+file, 
                                   Global.annarchy_dir+'/'+folder+'/'+file):
                    shutil.copy(Global.annarchy_dir+'/generate/'+folder+'/'+file, # src
                                Global.annarchy_dir+'/'+folder+'/'+file # dest
                    )
                    changed = True
                    if Global.config['verbose']:
                        Global._print(file, 'has changed since last compilation.')                        
            return changed
        
        def test_deleted(folder):
            changed = False
            # Copy the files which have changed 
            for file in os.listdir(Global.annarchy_dir+'/' + folder):
                # If the file does not exist in generate/build/ or generate/pyx/
                if not os.path.isfile(Global.annarchy_dir+'/generate/'+folder+'/'+file):
                    if not file.endswith('.o') and not file == 'ANNarchyCython.cpp': # skip generated files
                        os.remove(Global.annarchy_dir+'/'+folder+'/'+file)
                        if file.endswith('.cpp') : # suppress a population or projection
                            os.remove(Global.annarchy_dir+'/'+folder+'/'+file.replace('.cpp', '.o'))                            
                        changed = True
                        if Global.config['verbose']:
                            Global._print(file, 'has been deleted.') 
            return changed
        
        changed_cpp = False
        changed_pyx = False
        if not self.clean:
            import filecmp
            # Copy the files which have changed 
            changed_cpp = copy_changed('build')
            changed_pyx = copy_changed('pyx')
            # Test if some files have disappeared in generate
            changed_cpp = changed_cpp or test_deleted('build')
            changed_pyx = changed_pyx or test_deleted('pyx')
    
        else: # Copy everything
            for file in os.listdir(Global.annarchy_dir+'/generate/build'):
                shutil.copy(Global.annarchy_dir+'/generate/build/'+file, # src
                            Global.annarchy_dir+'/build/'+file # dest
                )
            for file in os.listdir(Global.annarchy_dir+'/generate/pyx'):
                shutil.copy(Global.annarchy_dir+'/generate/pyx/'+file, # src
                            Global.annarchy_dir+'/pyx/'+file # dest
                )
            changed_cpp = True
            changed_pyx = True
            
        return changed_cpp, changed_pyx
    
    def check_output(*popenargs, **kwargs):
        """Run command with arguments and return its output as a byte string.
         
        Backported from Python 2.7 as it's implemented as pure python on stdlib.
         
        >>> check_output(['/usr/bin/python', '--version'])
        Python 2.6.2

        adapted from: https://gist.github.com/edufelipe/1027906

        Modifications: Popen(cmd, stdout=...)
        """
        cmd = popenargs[1]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()

        if retcode:
            cmd = kwargs.get("args")

        if cmd is None:
            cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output

    def partial_compilation(self, changed_cpp, changed_pyx):
        """ Create ANNarchyCore.so and py extensions if something has changed."""
        if changed_cpp or changed_pyx:   
            Global._print('Compiling ... ')
            if Global.config['show_time']:
                t0 = time.time()
                                
        os.chdir(Global.annarchy_dir)
        if sys.platform.startswith('linux'): # Linux systems
            if not self.debug_build:
                flags = "-O2"
            else:
                flags = "-O0 -g -D_DEBUG"
    
            src = """# Makefile
SRC = $(wildcard build/*.cpp)
PYX = $(wildcard pyx/*.pyx)
OBJ = $(patsubst build/%.cpp, build/%.o, $(SRC))
     
all:
\t@echo "Please provide a target, either 'ANNarchyCython_2.6', 'ANNarchyCython_2.7' or 'ANNarchyCython_3.x for python versions."

ANNarchyCython_2.6: $(OBJ) pyx/ANNarchyCython_2.6.o
\t@echo "Build ANNarchyCython library for python 2.6"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.6.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.6 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.6.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ """+flags+""" -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.6 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.6.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_2.7: $(OBJ) pyx/ANNarchyCython_2.7.o
\t@echo "Build ANNarchyCython library for python 2.7"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_2.7.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython2.7 -o ANNarchyCython.so  

pyx/ANNarchyCython_2.7.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ """+flags+""" -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python2.7 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_2.7.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

ANNarchyCython_3.x: $(OBJ) pyx/ANNarchyCython_3.x.o
\t@echo "Build ANNarchyCython library for python 3.x"
\tg++ -shared -Wl,-z,relro -fpermissive -std=c++0x -fopenmp build/*.o pyx/ANNarchyCython_3.x.o -L. -L/usr/lib64 -Wl,-R./annarchy -lpython3.2mu -o ANNarchyCython.so  

pyx/ANNarchyCython_3.x.o : pyx/ANNarchyCython.pyx
\tcython pyx/ANNarchyCython.pyx --cplus  
\tg++ """+flags+""" -pipe -Wp,-D_FORTIFY_SOURCE=2 -fexceptions -fstack-protector --param=ssp-buffer-size=4 -D_GNU_SOURCE -fwrapv -fPIC -I/usr/include/python3.2 -c pyx/ANNarchyCython.cpp -o pyx/ANNarchyCython_3.x.o -L. -I. -Ibuild -fopenmp -std=c++0x -fpermissive 

build/%.o : build/%.cpp
\tg++ """+flags+""" -fPIC -pipe -fpermissive -std=c++0x -fopenmp -I. -c $< -o $@

ANNarchyCPP : $(OBJ)
\tg++ """+flags+""" -fPIC -shared -fpermissive -std=c++0x -fopenmp -I. build/*.o -o libANNarchyCPP.so

clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
    """
            
            # Write the Makefile to the disk
            with open('Makefile', 'w') as wfile:
                wfile.write(src)
            # Force recompilation of the Cython wrappers
            if changed_pyx: 
                os.system('touch pyx/ANNarchyCython.pyx')
            # Start the compilation
            try:
                if self.cpp_stand_alone:
                    subprocess.check_output('make ANNarchyCPP -j4 > compile_stdout.log 2> compile_stderr.log', 
                                            shell=True)
                elif sys.version_info[:2] == (2, 6):
                    self.check_output('make ANNarchyCython_2.6 -j4 > compile_stdout.log 2> compile_stderr.log', 
                                            shell=True)
                elif sys.version_info[:2] == (2, 7):
                    subprocess.check_output("make ANNarchyCython_2.7 -j4 > compile_stdout.log 2> compile_stderr.log", 
                                            shell=True)
                elif sys.version_info[:2] == (3, 2):
                    subprocess.check_output('make ANNarchyCython_3.x -j4 > compile_stdout.log 2> compile_stderr.log', 
                                            shell=True)
                else:
                    Global._error('No correct setup could be found. Do you have Python installed?')
                    exit(0)
            except subprocess.CalledProcessError:
                Global._error('Compilation failed.\nCheck the compilation logs in annarchy/compile_stderr.log')
                exit(0)
    
        else: # Windows: to test....
            sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')
            shutil.copy(sources_dir+'/compile.bat', Global.annarchy_dir)
            proc = subprocess.Popen(['compile.bat'], shell=True)
            proc.wait()
        
        Global._compiled = True
    
        if changed_cpp or changed_pyx:   
            Global._print('OK')
            if Global.config['show_time']:
                Global._print('Compilation took', time.time() - t0, 'seconds.')
                
    def instantiate(self):
        """ After every is compiled, actually create the Cython objects and 
            bind them to the Python ones."""
        # Return to the current directory
        os.chdir('..')
        # Import the Cython library
        try:
            import ANNarchyCython
        except ImportError, e:
            if not self.cpp_stand_alone:
                Global._print(e)
                Global._error('The Cython library was not correctly compiled.\n Check the compilation logs in annarchy/compile_sterr.log')
                exit(0)
        # Bind the py extensions to the corresponding python objects
        for pop in self.populations:
            if Global.config['verbose']:
                Global._print('    Create population', pop.name)
            if Global.config['show_time']:
                t0 = time.time()
            # Create the Cython instance 
            pop.cyInstance = eval('ANNarchyCython.py'+ pop.class_name+'()')
            # Create the attributes and actualize the initial values
            pop._init_attributes()
            if Global.config['show_time']:
                Global._print('Creating', pop.name, 'took', (time.time()-t0)*1000, 'milliseconds') 
                            
        # Instantiate projections
        for proj in self.projections:
            if Global.config['verbose']:
                Global._print('Creating projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"')        
            if Global.config['show_time']:
                t0 = time.time()
            # Create the synapses
            proj._connect() 
            
            #TODO:
            #if proj.connector.delays != None:
                #print 'set new delay',proj.connector.delays.max(),'for',proj.pre.name
            #    proj.pre.cyInstance.set_max_delay(int(proj.connector.delays.max()))
 
            # Create the attributes
            proj._init_attributes()   
            if Global.config['show_time']:
                Global._print('        took', (time.time()-t0)*1000, 'milliseconds')
                
    def create_includes(self):
        """ Generates 'Includes.h' containing all generated headers.
        """
        populations = self.analyser.analysed_populations.keys()
        projections = self.analyser.analysed_projections.keys()
        pop_header  = ''
        for pop in populations:
            pop_header += '#include "'+pop+'.h"\n'
    
        proj_header = ''
        for proj in projections:
            proj_header += '#include "'+ proj +'.h"\n'
    
        header = """#ifndef __ANNARCHY_INCLUDES_H__
#define __ANNARCHY_INCLUDES_H__
  
// population files
%(pop_header)s
// projection files
%(proj_header)s
#endif
""" % { 'pop_header': pop_header, 'proj_header': proj_header}
    
        with open(Global.annarchy_dir + '/generate/build/Includes.h', mode = 'w') as w_file:
            w_file.write(header)
            
    def update_annarchy_header(self):
        """ Updates ANNarchy.h depending on compilation modes:
        
        *Available modes*:
         
            * cpp_stand_alone::
            
                True: instantiation of population, projection classes and projection instantiation.
                False: only projection instantiation.
    
            * profile_enabled::
            
                True: enabled profile
                False: disabled profile
        """
        code = ''
        with open(Global.annarchy_dir+'/generate/build/ANNarchy.h', mode = 'r') as r_file:
            for a_line in r_file:
                if a_line.find('//AddProjection') != -1:
                    if(self.cpp_stand_alone):
                        pass #TODO
#                         for proj in projections:
#                             code += proj.generator.generate_cpp_add()
                elif a_line.find('//AddPopulation') != -1:
                    if(self.cpp_stand_alone):
                        pass # TODO
#                         for pop in populations:
#                             code += pop.generator.generate_cpp_add()                            
                elif a_line.find('//createProjInstance') != -1:
                    code += a_line
                    code += self.generate_proj_instance_class()
                else:
                    code += a_line
    
        with open(Global.annarchy_dir+'/generate/build/ANNarchy.h', mode='w') as w_file:
            w_file.write(code)
            

    def generate_proj_instance_class(self):
        """
        The standard ANNarchy core has no knowledge about the full amount of projection
        classes. On the other hand we want to instantiate the object from there. To achieve this
        we introduce a projection instantiation class, which returns a projections instance corresponding
        to the given ID.
        """
        projections = self.analyser.analysed_projections
        # single cases
        cases_ptr = ''
        for name, desc in projections.iteritems():
            cases_ptr += """
                case %(id)s:
                    {
                #ifdef _DEBUG
                    std::cout << "Instantiate name=%(name)s" << std::endl;
                #endif
                    return new %(name)s(pre, post, postNeuronRank, target, spike);
                    }

""" % { 'name': name, 'id': name.split('Projection')[1]}

        # complete code
        code = """class createProjInstance {
public:
    createProjInstance() {};
    
    /**
     *    @brief      instantiate a projection object or returns previous exsisting one.
     *    @details    called by cpp method ANNarchy::ANNarchy() or by 
     *                createProjInstance::getInstanceOf(int, int, int, int, int)
     */
    Projection* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target, bool spike) {
        
        if(pre == NULL || post == NULL) {
            std::cout << "Critical error: invalid pointer in c++ core library." << std::endl;
            return NULL;
        }
        
        // search for already existing instance
        Projection* proj = post->getProjection(postNeuronRank, target, pre);
        
        if(proj)
        {
            // return existing one
            return proj;
        }
        else
        {
            switch(ID) 
            {
%(case1)s
                default:
                {
                    std::cout << "Unknown typeID: "<< ID << std::endl;
                    return NULL;
                }
            }                    
        }
    }

    /**
     *  @brief          instantiate a projection object or returns previous exsisting one.
     *  @details        called by cython wrapper.
     */
    Projection* getInstanceOf(int ID, int preID, int postID, int postNeuronRank, int target, bool spike) {
        Population *pre  = Network::instance()->getPopulation(preID);
        Population *post = Network::instance()->getPopulation(postID);
        
        return getInstanceOf(ID, pre, post, postNeuronRank, target, spike);
    }

};
""" % { 'case1': cases_ptr }
        return code
    
    def update_global_header(self):
        """
        update Global.h dependent on compilation modes:
        
        *available modes*:
    
            * profile_enabled::
            
                True: enabled profile
                False: disabled profile
        """
        code = ''
        
        with open(Global.annarchy_dir+'/generate/build/Global.h', mode = 'r') as r_file:
            for a_line in r_file:
                if (a_line.find('ANNAR_PROFILE') != -1 and 
                    a_line.find('define') != -1):
                    if self.profile_enabled:
                        code += '#define ANNAR_PROFILE\n'
                elif a_line.find('//FUNCTIONS') != -1: # Put the global functions here
                    code += a_line
                    for func in Global._functions:
                        code += "    " +  _extract_functions(func, local_global=True)[0]['cpp']
                else:
                    code += a_line
    
        with open(Global.annarchy_dir+'/generate/build/Global.h', mode = 'w') as w_file:
            w_file.write(code)
                


    def generate_py_extension(self):
        """
        Because the amount of code is higher, we decide to split up the code. Nevertheless cython generates 
        one shared library per .pyx file. To retrieve only one library we need to compile only one .pyx file
        which includes all the others. 
        """
        populations = self.analyser.analysed_populations.keys()
        projections = self.analyser.analysed_projections.keys()
           
        pop_include = ''
        for pop in populations:
            pop_include += 'include \"'+pop+'.pyx\"\n'
    
        proj_include = ''
        for proj in projections:
            proj_include += 'include \"'+proj+'.pyx\"\n'
    
        code = """include "Network.pyx"

%(pop_inc)s  

include "Projection.pyx"
%(proj_inc)s  

%(profile)s
include "Connector.pyx"
""" % { 'pop_inc': pop_include,
        'proj_inc': proj_include,
        'profile': 'include "cy_profile.pyx"' if self.profile_enabled else '' 
        }
    
        with open(Global.annarchy_dir+'/generate/pyx/ANNarchyCython.pyx', mode='w') as w_file:
            w_file.write(code)
        
