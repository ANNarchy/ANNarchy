""" 

    DendriteGenerator.py
    
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
from ANNarchy.core.Random import *

from Templates import *

import re

class DendriteGenerator(object):
    """ Base class for generating C++ code from a synapse description. """
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.name+'.cpp'
        
    def generate(self, verbose):
        self.verbose = verbose
        if verbose:
            Global._print( 'Generating', self.name )

        with open(self.header, mode = 'w') as w_file:
            w_file.write(self.generate_header())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self.generate_body())

    def generate_members_declaration(self):
        """ 
        Returns private members declaration. 
        
        Notes:            
            * depend only on locality
            * value should not be defined twice.
        """
        members = ""
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue

            if param['name'] == 'value': # the vector is already declared
                members += """
    // value_ : local
    bool record_%(name)s_; 
    std::vector< std::vector<%(type)s> > recorded_%(name)s_;    
""" % {'name' : param['name'], 'type': param['ctype']}

            elif param['name'] in self.desc['local']: # local attribute
                members += """
    // %(name)s_ : local
    std::vector<%(type)s> %(name)s_;  
    bool record_%(name)s_; 
    std::vector< std::vector<%(type)s> > recorded_%(name)s_;    
""" % {'name' : param['name'], 'type': param['ctype']}

            elif param['name'] in self.desc['global']: # global attribute
                members += """
    // %(name)s_ : global
    %(type)s %(name)s_;   
""" % {'name' : param['name'], 'type': param['ctype']}

        return members
    
    def generate_members_access(self):
        """ 
        Returns public members access. 
        
        Notes:
            * depend only on locality.
            * rank, delay access is already implemented in base class
            * psp are skipped, hence it should not be accessible 
        """
        members = ""
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "value":
                func = """
    std::vector< std::vector< %(type)s > > getRecorded%(Name)s() { return this->recorded_%(name)s_; }                    
    void startRecord%(Name)s() { this->record_%(name)s_ = true; }
    void stopRecord%(Name)s() { this->record_%(name)s_ = false; }
    void clearRecorded%(Name)s() { this->recorded_%(name)s_.clear(); }
"""
                members += func % { 'name' : param['name'], 
                                    'Name': param['name'].capitalize(),
                                    'type': param['ctype']}
            elif param['name'] in self.desc['local']: # local attribute
                members += local_variable_access % { 'name' : param['name'], 
                                                     'Name': param['name'].capitalize(),
                                                     'type': param['ctype']}
                
            elif param['name'] in self.desc['global']: # global attribute
                members += global_variable_access % { 'name' : param['name'], 
                                                      'Name': param['name'].capitalize(),
                                                      'type': param['ctype']}

        return members

    def generate_constructor(self):
        """ Content of the Projection constructor."""
        constructor = ""
        # Attributes
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == "value":
                continue
            elif param['name'] in self.desc['local']: # local attribute
                ctype = param['ctype']
                if ctype == 'bool':
                    cinit = 'true' if param['init'] else 'false'
                elif ctype == 'int':
                    cinit = int(param['init'])
                elif ctype == 'DATA_TYPE':
                    cinit = float(param['init'])
                constructor += """
    // %(name)s_ : local
    %(name)s_ = std::vector<%(type)s> ( rank_.size(), %(init)s);  
    record_%(name)s_ = false; 
""" % {'name' : param['name'], 'type': param['ctype'], 'init' : str(cinit)}

            elif param['name'] in self.desc['global']: # global attribute
                ctype = param['ctype']
                if ctype == 'bool':
                    cinit = 'true' if param['init'] else 'false'
                elif ctype == 'int':
                    cinit = int(param['init'])
                elif ctype == 'DATA_TYPE':
                    cinit = float(param['init'])
                constructor += """
    // %(name)s_ : global
    %(name)s_ = %(init)s;   
""" % {'name' : param['name'], 'init': str(cinit)}   

        constructor += '\n    // Time step dt_\n    dt_ = ' + str(Global.config['dt']) + ';\n'
        return constructor 
        
    def generate_destructor(self):
        """ Content of the Projection destructor."""
        return ""
    
    def generate_cwrappers(self):
        """
        Parts of the C++ header which are exported to Python through Cython.
        
        Notes:
            * each exported function need a wrapper method, in this function the wrappers for variables and parameters are generated. 
            * the access to GPU data is only possible through host. Through this detail the implementation of the cython wrappers are equal to all paradigms.
        """
        code = ""

        local_template = Templates.local_wrapper_pyx
        global_template = Templates.global_wrapper_pyx
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == "value":
                continue
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_template % { 'Name': param['name'].capitalize(), 
                                              'name': param['name'], 
                                              'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
                
            elif param['name'] in self.desc['global']: # global attribute
                code += global_template % { 'Name': param['name'].capitalize(), 
                                               'name': param['name'], 
                                               'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
        
        return code

    def generate_pyfunctions(self):
        """
        Python functions accessing the Cython wrapper
        
        Notes:
            * the access to GPU data is only possible through host. Through this detail the implementation of the cython wrappers are equal to all paradigms.
        """
        code = ""        

        local_template = Templates.local_property_pyx
        global_template = Templates.global_property_pyx
         
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == "value":
                continue
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_template % { 'Name': param['name'].capitalize(), 
                                           'name': param['name'],
                                           'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float' }
            elif param['name'] in self.desc['global']: # global attribute
                code += global_template % { 'Name': param['name'].capitalize(), 
                                            'name': param['name'],
                                            'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float' }
        return code

        
    def generate_record(self):
        """ 
        Code for recording.

        Notes:
            * only local variables / parameters are recorded.
        """
        code = ""
        # Attributes
        for param in list(set(self.desc['local'] + ['value'])): # local attribute
            code += """
    if(record_%(var)s_)
    {
        recorded_%(var)s_.push_back(%(var)s_);
    }
""" % { 'var': param }

        return code
  
    def generate_add_synapse(self):
        """
        Code for add synapses.
        
        Notes:
            * the implemented template is only extended by local variables / parameters
            * value and delay are skipped.
        """
        code = ""
        
        for var in self.desc['variables'] + self.desc['parameters']:
            if var['name'] == 'value' or var['name'] == 'delay':
                continue

            if var['name'] in self.desc['local']:
                code += """%(var)s_.push_back(%(init)s); """ % { 'var': var['name'], 'init': var['init'] }
            
        return add_synapse_body % { 'add_synapse': code }

    def generate_rem_synapse(self):
        """
        Code for remove synapses.
        
        Notes:
            * the implemented template is only extended by local variables / parameters
            * value and delay are skipped.
        """
        code = ""
        
        for var in self.desc['variables'] + self.desc['parameters']:
            if var['name'] == 'value'or var['name'] == 'delay':
                continue

            if var['name'] in self.desc['local']:
                code += """%(var)s_.erase(%(var)s_.begin()+i);""" % { 'var' : var['name'] }

        return rem_synapse_body % { 'rem_synapse': code }

    def generate_rem_all_synapse(self):
        """
        Code for remove all synapses.
        
        Notes:
            * the implemented template is only extended by local variables / parameters
            * value and delay are skipped.
        """
        code = ""
        
        for var in self.desc['variables'] + self.desc['parameters']:
            if var['name'] == 'value'or var['name'] == 'delay':
                continue

            if var['name'] in self.desc['local']:
                code += """%(var)s_.clear();""" % { 'var' : var['name'] }

        return rem_all_synapse_body % { 'rem_all_synapse': code }
    
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
    