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
from ANNarchy4.core.Descriptor import Attribute   

def create_includes():
    """
    generate 'Includes.h' containing all generated headers.
    """
    pop_header  = ''
    for pop in Global._populations:
        pop_header += '#include "'+pop.generator.class_name+'.h"\n'

    proj_header = ''
    for proj in Global._projections:
        proj_header += '#include "'+ proj.generator.proj_class['name'] +'.h"\n'

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

def update_annarchy_header(cpp_stand_alone, profile_enabled):
    """
    update ANNarchy.h dependent on compilation modes:
    
    *available modes*:
     
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
                if(cpp_stand_alone):
                    for proj in Global._projections:
                        code += proj.generator.generate_cpp_add()
            elif a_line.find('//AddPopulation') != -1:
                if(cpp_stand_alone):
                    for pop in Global._populations:
                        code += pop.generator.generate_cpp_add()
                        
            elif a_line.find('//createProjInstance') != -1:
                code += a_line
                code += generate_proj_instance_class()
            else:
                code += a_line

    with open(Global.annarchy_dir+'/generate/build/ANNarchy.h', mode='w') as w_file:
        w_file.write(code)

def update_global_header(profile_enabled):
    """
    update Global.h dependent on compilation modes:
    
    *available modes*:

        * profile_enabled::
        
            True: enabled profile
            False: disabled profile
    """
    code = ''
    
    if profile_enabled:
        with open(Global.annarchy_dir+'/generate/build/Global.h', mode = 'r') as r_file:
            for a_line in r_file:
                if (a_line.find('ANNAR_PROFILE') != -1 and 
                    a_line.find('define') != -1):
                    code += '#define ANNAR_PROFILE\n'
                else:
                    code += a_line

        with open(Global.annarchy_dir+'/generate/build/Global.h', mode = 'w') as w_file:
            w_file.write(code)

def generate_proj_instance_class():
    """
    The standard ANNarchy core has no knowledge about the full amount of projection
    classes. On the other hand we want to instantiate the object from there. To achieve this
    we introduce a projection instantiation class, which returns a projections instance corresponding
    to the given ID.
    """
    # single cases
    cases_ptr = ''
    for proj in Global._projections:
        cases_ptr += """
            case %(id)s:
                {
            #ifdef _DEBUG
                std::cout << "Instantiate name=%(name)s and id=%(id)s" << std::endl;
            #endif
                return new %(name)s(pre, post, postNeuronRank, target);
                }

""" % { 'id': proj.generator.proj_class['ID'], 
        'name': proj.generator.proj_class['name']
    }

    # complete code
    code = """class createProjInstance {
public:
    createProjInstance() {};
    
    /**
     *    @brief      instantiate a projection object or returns previous exsisting one.
     *    @details    called by cpp method ANNarchy::ANNarchy() or by 
     *                createProjInstance::getInstanceOf(int, int, int, int, int)
     */
    Projection* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target) {
        
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
    Projection* getInstanceOf(int ID, int preID, int postID, int postNeuronRank, int target) {
        Population *pre  = Network::instance()->getPopulation(preID);
        Population *post = Network::instance()->getPopulation(postID);
        
        return getInstanceOf(ID, pre, post, postNeuronRank, target);
    }

};
""" % { 'case1': cases_ptr }
    return code

def generate_py_extension(profile_enabled):
    """
    Hence the amount of code is higher, we decide to split up the code. Nevertheless cython generates 
    one shared library per .pyx file. To retrieve only one library we need to compile only one .pyx file
    which includes all the others. 
    """
    pop_include = ''
    for pop in Global._populations:
        pop_include += 'include \"'+pop.generator.class_name+'.pyx\"\n'

    proj_include = ''
    for proj in Global._projections:
        proj_include += 'include \"'+proj.generator.proj_class['name']+'.pyx\"\n'

    code = """include "Network.pyx"

%(pop_inc)s  

include "Projection.pyx"
%(proj_inc)s  

%(profile)s
include "Connector.pyx"
""" % { 'pop_inc': pop_include,
        'proj_inc': proj_include,
        'profile': 'include "Profile.pyx"' if profile_enabled else '' 
    }

    with open(Global.annarchy_dir+'/generate/pyx/ANNarchyCython.pyx', mode='w') as w_file:
        w_file.write(code)
        
def folder_management(profile_enabled, clean):
    """
    ANNarchy is provided as a python package. For compilation a local folder
    'annarchy' is created in the current working directory.
    
    *Parameter*:
    
    * *profile_enabled*: copy needed data for profile extension
    """
    sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')
    if os.path.exists(Global.annarchy_dir): # Already compiled
        if clean:
            shutil.rmtree(Global.annarchy_dir, True)
    else:    
        os.mkdir(Global.annarchy_dir)
        os.mkdir(Global.annarchy_dir+'/pyx')
        os.mkdir(Global.annarchy_dir+'/build')
        os.mkdir(Global.annarchy_dir+'/generate')
        os.mkdir(Global.annarchy_dir+'/generate/pyx')
        os.mkdir(Global.annarchy_dir+'/generate/build')

            
    # cpp / h files
    for file in os.listdir(sources_dir+'/cpp'):
        shutil.copy(sources_dir+'/cpp/'+file, # src
                    Global.annarchy_dir+'/generate/build/'+file # dest
                    )
    # pyx files
    for file in os.listdir(sources_dir+'/pyx'):
        shutil.copy(sources_dir+'/pyx/'+file, #src
                    Global.annarchy_dir+'/generate/pyx/'+file #dest
                    )
    # profile files
    if profile_enabled:
        profile_sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../extensions/Profile')
    
        shutil.copy(profile_sources_dir+'/Profile.cpp', Global.annarchy_dir+'/generate/build')
        shutil.copy(profile_sources_dir+'/Profile.h', Global.annarchy_dir+'/generate/build')
        shutil.copy(profile_sources_dir+'/Profile.pyx', Global.annarchy_dir+'/generate/pyx')

    sys.path.append(Global.annarchy_dir)

def _update_float_prec(file):
    """
    all code generated files will be updated
    """
    code = ''
    with open(file, mode = 'r') as r_file:
        for a_line in r_file:
            if Global.config['float_prec']=='single':
                code += a_line.replace("DATA_TYPE","float")
            else:
                code += a_line.replace("float","double").replace("DATA_TYPE","double")

    with open(file, mode='w') as w_file:
        w_file.write(code)
    
def code_generation(cpp_stand_alone, profile_enabled, clean):
    """
    code generation for each population respectively projection object the user defined. 
    
    After this the ANNarchy main header is expanded by the corresponding headers.
    """
    if Global.config['verbose']:
        print '\nGenerate code ...'

    # create population cpp class for each neuron
    for pop in Global._populations:
        pop.generator.generate(Global.config['verbose'])

    # create projection cpp class for each synapse
    for projection in Global._projections:
        projection.generator.generate(Global.config['verbose'])

    # Create includes
    create_includes()

    if Global.config['verbose']:
        print '\nUpdate ANNarchy header ...'
    update_annarchy_header(cpp_stand_alone, profile_enabled)

    if Global.config['verbose']:
        print '\nUpdate global header ...'
    update_global_header(profile_enabled)

    if Global.config['verbose']:
        print '\nGenerate py extensions ...'
    generate_py_extension(profile_enabled)
    
    os.chdir(Global.annarchy_dir+'/generate/build')
    cpp_src = filter(os.path.isfile, os.listdir('.'))
    for file in cpp_src:
        _update_float_prec(file)
            
    os.chdir(Global.annarchy_dir+'/generate/pyx')
    pyx_src = filter(os.path.isfile, os.listdir('.'))
    for file in pyx_src:
        _update_float_prec(file)
    os.chdir(Global.annarchy_dir)    
    
    # Test if the code generation has changed
    changed_cpp = False
    changed_pyx = False
    if not clean:
        import filecmp
        for file in os.listdir(Global.annarchy_dir+'/generate/build'):
            # If the file does not exist in build/, or if the newly generated file is different, copy the new file
            if not os.path.isfile(Global.annarchy_dir+'/build/'+file) or not filecmp.cmp(Global.annarchy_dir+'/generate/build/'+file, Global.annarchy_dir+'/build/'+file):
                shutil.copy(Global.annarchy_dir+'/generate/build/'+file, # src
                            Global.annarchy_dir+'/build/'+file # dest
                )
                changed_cpp = True
                if Global.config['verbose']:
                    print file, 'has changed since last compilation.'
        for file in os.listdir(Global.annarchy_dir+'/generate/pyx'):
            if not os.path.isfile(Global.annarchy_dir+'/pyx/'+file) or not filecmp.cmp(Global.annarchy_dir+'/generate/pyx/'+file, Global.annarchy_dir+'/pyx/'+file):
                shutil.copy(Global.annarchy_dir+'/generate/pyx/'+file, # src
                            Global.annarchy_dir+'/pyx/'+file # dest
                )
                changed_pyx = True
                if Global.config['verbose']:
                    print file, 'has changed since last compilation.'
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
    
def _update_global_operations():
    
    for proj in Global._projections:
        if proj.synapse != None:
            proj_dict = proj.synapse._global_operations()
            
            # only post-synaptic variables are generated in populations
            for entry in proj_dict['pre']:
                #print 'Add to', proj.pre.name,'the item', entry
                proj.pre.generator._add_global_oparation(entry) 
            
            for entry in proj_dict['post']:
                #print 'Add to', proj.post.name,'the item', entry
                proj.post.generator._add_global_oparation(entry)
            
    
def compile(clean=False, cpp_stand_alone=False, debug_build=False):
    """
    This method uses the network architecture to generate optimized C++ code and compile a shared library that will carry the simulation.
    
    *Parameters*:

    * *clean*: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
    * *cpp_stand_alone*: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
    * *debug_build*: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    """
    print 'ANNarchy', ANNarchy4.__version__, 'on', sys.platform, '(', os.name,')'
        
    # Test if profiling is enabled
    profile_enabled = False
    try:
        from ANNarchy4.extensions import Profile
    except ImportError:
        pass
    else:
        profile_enabled = True
        
    # Create the necessary subfolders and copy the source files
    if Global.config['verbose']:
        print "Create 'annarchy' subdirectory."
    folder_management(profile_enabled, clean)
    
    # Tell each population which global operation they should compute
    _update_global_operations()
    
    # Generate the code
    changed_cpp, changed_pyx = code_generation(cpp_stand_alone, profile_enabled, clean)
    
    # Create ANNarchyCore.so and py extensions 
    if changed_cpp or changed_pyx:   
        print 'Compiling ...'
    if Global.config['show_time']:
        t0 = time.time()
            
    os.chdir(Global.annarchy_dir)
    if sys.platform.startswith('linux'):
        if not debug_build:
            flags = "-O2"
        else:
            flags = "-O0 -g -D_DEBUG"
        src = """# Makefile
SRC = $(wildcard build/*.cpp)
PYX = $(wildcard pyx/*.pyx)
OBJ = $(patsubst build/%.cpp, build/%.o, $(SRC))
     
all:
\t@echo "Please provide a target, either 'ANNarchyCython_2.6', 'ANNarchyCython_2.7' or 'ANNarchyCython_3.x"

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

clean:
\trm -rf build/*.o
\trm -rf pyx/*.o
\trm -rf ANNarchyCython.so
"""

        with open('Makefile', 'w') as wfile:
            wfile.write(src)
        if changed_pyx: # Force recompilation of the Cython wrappers
            os.system('touch pyx/ANNarchyCython.pyx')

        if sys.version_info[:2] == (2, 6):
            os.system('make ANNarchyCython_2.6 -j4 > compile_stdout.log 2> compile_stderr.log')
        elif sys.version_info[:2] == (2, 7):
            os.system('make ANNarchyCython_2.7 -j4 > compile_stdout.log 2> compile_stderr.log')
        else:
            os.system('make ANNarchyCython_3.x -j4 > compile_stdout.log 2> compile_stderr.log')
        
    else: # Windows: to test....
        sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')
        shutil.copy(sources_dir+'/compile.bat', Global.annarchy_dir)
        proc = subprocess.Popen(['compile.bat'], shell=True)
        proc.wait()
    
    Global._compiled = True
    if Global.config['show_time']:
        print 'Compilation took', time.time() - t0, 'seconds.'
        
    if Global.config['verbose']:
        print 'Building network ...' 
          
    # Return to the current directory
    os.chdir('..')
    # Import the libraries
    try:
        import ANNarchyCython
    except:
        if not cpp_stand_alone:
            print '\nError: the Cython library was not correctly compiled.'
            exit(0)
            
    # Create the Python objects    
    if not cpp_stand_alone:
        # bind the py extensions to the corresponding python objects
        for pop in Global._populations:
            if Global.config['verbose']:
                print '    Create population', pop.name
            if Global.config['show_time']:
                t0 = time.time()
            # Create the Cython instance
            pop.cyInstance = eval('ANNarchyCython.py'+ pop.generator.class_name+'()')
            # Create the attributes
            pop._init_attributes()
            # Initialize their value
            pop.generator._init_variables()
            if Global.config['show_time']:
                print 'Creating', pop.name, 'took', (time.time()-t0)*1000, 'milliseconds'             
        # instantiate projections
        for proj in Global._projections:
            if Global.config['verbose']:
                print 'Creating projection from', proj.pre.name,'to', proj.post.name,'with target="', proj.target,'"'           
            if Global.config['show_time']:
                t0 = time.time()
            # Create the synapses
            proj.connect() 
            if proj.connector.delays != None:
                #print 'set new delay',proj.connector.delays.max(),'for',proj.pre.name
                proj.pre.cyInstance.set_max_delay(int(proj.connector.delays.max()))
 
            # Create the attributes
            proj._init_attributes()   
            if Global.config['show_time']:
                print '        took', (time.time()-t0)*1000, 'milliseconds'  
#        print 'OK.'           

    else:
        #abort the application after compiling ANNarchyCPP
        print '\nCompilation process of ANNarchyCPP completed successful.\n'
        exit(0)
