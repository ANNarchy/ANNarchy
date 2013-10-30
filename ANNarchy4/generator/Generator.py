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
import exceptions

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

    with open(Global.annarchy_dir + '/build/Includes.h', mode = 'w') as w_file:
        w_file.write(header)

def update_annarchy_header(cpp_stand_alone):
    """
    update ANNarchy.h dependent on compilation mode (cpp_stand_alone):
        - True: instantiation of population, projection classes and projection instantiation.
        - False: only projection instantiation.
    """
    code = ''
    with open(Global.annarchy_dir+'/build/ANNarchy.h', mode = 'r') as r_file:
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

    with open(Global.annarchy_dir+'/build/ANNarchy.h', mode='w') as w_file:
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
            return new %(name)s(pre, post, postNeuronRank, target);

""" % { 'id': proj.generator.proj_class['ID'], 
        'name': proj.generator.proj_class['name']
    }

    cases_id = ''
    for proj in Global._projections:
        cases_id += """
        case %(id)s:
        #ifdef _DEBUG
            std::cout << "Instantiate name=%(name)s and id=%(id)s" << std::endl;
        #endif
            return new %(name)s(preID, postID, postNeuronRank, target);

""" % { 'id': proj.generator.proj_class['ID'], 
        'name': proj.generator.proj_class['name']
    }

    # complete code
    code = """class createProjInstance {
public:
    createProjInstance() {};

    Projection* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target) {
        switch(ID) {
%(case1)s
            default:
                std::cout << "Unknown typeID: "<< ID << std::endl;
                return NULL;
        }
    }

    Projection* getInstanceOf(int ID, int preID, int postID, int postNeuronRank, int target) {
        switch(ID) {
%(case2)s
            default:
                std::cout << "Unknown typeID: "<< ID << std::endl;
                return NULL;
        }
    }

};
""" % { 'case1': cases_ptr, 'case2': cases_id }
    return code

def generate_py_extension():
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

include "Connector.pyx"
""" % { 'pop_inc': pop_include,
        'proj_inc': proj_include }

    with open(Global.annarchy_dir+'/pyx/ANNarchyCython.pyx', mode='w') as w_file:
        w_file.write(code)
        
def folder_management():
    """
    ANNarchy is provided as a python package. For compilation a local folder
    'annarchy' is created in the current working directory.
    """
    if os.path.exists(Global.annarchy_dir):
        shutil.rmtree(Global.annarchy_dir, True)
    
    os.mkdir(Global.annarchy_dir)
    os.mkdir(Global.annarchy_dir+'/pyx')
    os.mkdir(Global.annarchy_dir+'/build')

    sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')

    for file in os.listdir(sources_dir):
        if not os.path.isdir(os.path.abspath(sources_dir+'/'+file)):
            shutil.copy(sources_dir+'/'+file, Global.annarchy_dir)
            
    for file in os.listdir(sources_dir+'/cpp'):
        shutil.copy(sources_dir+'/cpp/'+file, # src
                    Global.annarchy_dir+'/build/'+file # dest
                    )
        
    for file in os.listdir(sources_dir+'/pyx'):
        shutil.copy(sources_dir+'/pyx/'+file, #src
                    Global.annarchy_dir+'/pyx/'+file #dest
                    )

    sys.path.append(Global.annarchy_dir)

def code_generation(cpp_stand_alone, verbose):
    """
    code generation for each population respectively projection object the user defined. 
    
    After this the ANNarchy main header is expanded by the corresponding headers.
    """
    if verbose:
        print '\nGenerate code ...'

    # create population cpp class for each neuron
    for pop in Global._populations:
        try:
            pop.generator.generate(verbose)
        except exceptions.TypeError:
            print 'Error on code generation for',pop.name
            return
        
    # create projection cpp class for each synapse
    for projection in Global._projections:
        projection.generator.generate(verbose)

    create_includes()

    if verbose:
        print '\nUpdate ANNarchy header ...'
    update_annarchy_header(cpp_stand_alone)

    if verbose:
        print '\nGenerate py extensions ...'
    generate_py_extension()
    
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
            
    
def compile(verbose=False, cpp_stand_alone=False, debug_build=False):
    """
    The compilation procedure consists roughly of 3 steps:
    
        a) generate user defined classes and cython wrapper
        b) compile ANNarchyCore
        c) compile ANNarchyCython
        
    after this the cythonized objects are instantiated and available for the user. 
    
    *Parameters*:

        * *verbose*: shows details about compilation process on console (by default False).
        * *cpp_stand_alone*: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
        * *debug_build*: creates a debug version of ANNarchy, which logs the creation of objects and some other data (by default False).
    """
    print 'ANNarchy', ANNarchy4.__version__, 'on', sys.platform, '(', os.name,')'
    # Create the necessary subfolders and copy the source files
    if verbose:
        print "\nCreate 'annarchy' subdirectory."
    folder_management()
    # Tell each population which global operation they should compute
    _update_global_operations()
    # Generate the code
    code_generation(cpp_stand_alone, verbose)
    
    # Create ANNarchyCore.so and py extensions
    print '\nCompiling...',
    os.chdir(Global.annarchy_dir)
    # Make sure the makefiles are executable
    if sys.platform.startswith('linux'):
        os.system('chmod +x compile*')
    # Start the compilation depending on the platform
    if not debug_build:
        if sys.platform.startswith('linux'):
            proc = subprocess.Popen(['./compile.sh', 'cpp_stand_alone='+str(cpp_stand_alone)])
        else:
            proc = subprocess.Popen(['compile.bat'], shell=True)
        proc.wait()
    else:
        if sys.platform.startswith('linux'):
            proc = subprocess.Popen(['./compiled.sh', 'cpp_stand_alone='+str(cpp_stand_alone)])
        else:
            proc = subprocess.Popen(['compile.bat'], shell=True)
        proc.wait()
        
    Global._compiled = True
    
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
            # Create the Cython instance
            pop.cyInstance = eval('ANNarchyCython.py'+ pop.generator.class_name+'()')
            # Create the attributes
            pop._init_attributes()
            # Initialize their value
            pop.generator._init_variables()
        # instantiate projections
        for proj in Global._projections:
            # Create the synapses
            proj.connect()  
            # Create the attributes
            proj._init_attributes()
        print ' OK.'   
    else:
        #abort the application after compiling ANNarchyCPP
        print '\nCompilation process of ANNarchyCPP completed successful.\n'
        exit(0)
                
    
