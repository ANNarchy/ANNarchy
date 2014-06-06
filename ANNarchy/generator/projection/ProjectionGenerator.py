""" 

    ProjectionGenerator.py
    
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
from ANNarchy.core import Global

from Templates import *

import re

class ProjectionGenerator(object):
    """ Base class for generating C++ code from a population description. """
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        
        # Names of the files to be generated
        self.proj_header = Global.annarchy_dir+'/generate/build/'+self.name+'.h'
        self.proj_body = Global.annarchy_dir+'/generate/build/'+self.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.name+'.pyx'

    def generate(self, verbose):
        self.verbose = verbose
        if verbose:
            Global._print( 'Generating', self.name )

        #   generate files
        with open(self.proj_header, mode = 'w') as w_file:
            w_file.write(self.generate_header())

        with open(self.proj_body, mode = 'w') as w_file:
            w_file.write(self.generate_body())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generate_pyx()) 

    def generate_dendrite_access_declaration(self):
        """ 
        Returns access methods towards the attached dendrites 
        
        Notes:
            * depend only on locality.
            * rank, delay access is already implemented in base class
            * w is partly predefined
            * the definition of the function is implemented in the body 
        """
        members_header = ""
        
        local_template = local_idx_variable_access
        global_template = global_idx_variable_access
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "w":
                func = """
    std::vector< std::vector< %(type)s > >get_recorded_%(name)s(int post_rank);                    
    void start_record_%(name)s(int post_rank);
    void stop_record_%(name)s(int post_rank);
    void clear_recorded_%(name)s(int post_rank);
"""             
                members_header += func % { 'name' : param['name'], 
                                           'type': param['ctype'],
                                           'class': self.name.replace('Projection', 'Dendrite')}
            elif param['name'] in self.desc['local']: # local attribute
                members_header += local_template % { 'name' : param['name'], 
                                              'type': param['ctype'],
                                              'class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['global']: # global attribute
                members_header += global_template % { 'name' : param['name'], 
                                               'type': param['ctype'],
                                               'class': self.name.replace('Projection', 'Dendrite')}

        return members_header

    def generate_dendrite_access_definition(self):
        """ 
        Returns implementation of attached dendritic data  
        
        Notes:
            * depend only on locality.
            * rank, delay access is already implemented in base class
            * psp are skipped, hence it should not be accessible 
        """
        members_body = ""
        
        local_template = local_idx_variable_access_body
        global_template = global_idx_variable_access_body
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "w":
                func = """
std::vector< std::vector< %(type)s > > %(class)s::get_recorded_%(name)s(int post_rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_recorded_%(name)s(); 
}                    
void %(class)s::start_record_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->start_record_%(name)s(); 
}
void %(class)s::stop_record_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->stop_record_%(name)s(); 
}
void %(class)s::clear_recorded_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->clear_recorded_%(name)s(); 
}
"""             
                members_body += func % { 'name' : param['name'], 
                                         'type': param['ctype'],
                                         'class': self.name,
                                         'dend_class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['local']: # local attribute
                members_body += local_template % { 'name' : param['name'], 
                                              'type': param['ctype'],
                                              'class': self.name,
                                              'dend_class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['global']: # global attribute
                members_body += global_template % { 'name' : param['name'], 
                                               'type': param['ctype'],
                                               'class': self.name,
                                               'dend_class': self.name.replace('Projection', 'Dendrite')}

        return members_body
    
    def generate_cwrappers(self):
        """
        Parts of the C++ header which are exported to Python through Cython.
        
        Notes:
            * each exported function need a wrapper method, in this function the wrappers for variables and parameters are generated. 
            * the access to GPU data is only possible through host. Through this detail the implementation of the cython wrappers are equal to all paradigms.
        """
        code = ""
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == "w":
                continue
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_wrapper_pyx % { 'name': param['name'], 
                                              'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
                
            elif param['name'] in self.desc['global']: # global attribute
                code += global_wrapper_pyx % { 'name': param['name'], 
                                               'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
        
        return code

    def generate_pyfunctions(self):
        """
        Python functions accessing the Cython wrapper
        
        Notes:
            * the access to GPU data is only possible through host. Through this detail the implementation of the cython wrappers are equal to all paradigms.
        """
        code = ""        
         
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == "w":
                continue
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_property_pyx % { 'name': param['name'],
                                               'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float' }
            elif param['name'] in self.desc['global']: # global attribute
                code += global_property_pyx % { 'name': param['name'],
                                                'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float' }
        return code

  
    def generate_functions(self):
        "Custom functions"
        code = ""        
        for func in self.desc['functions']:
            
            code += '    ' + func['cpp']
            
        return code

    def generate_add_proj_include(self):
        """
        Include of paradigm specific headers
        """
        add_include = ''
        if Global.config['paradigm'] == "cuda":
            add_include += "#include \"simple_test.h\""
        
        return add_include
    
