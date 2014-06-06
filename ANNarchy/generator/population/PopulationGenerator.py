""" 

    PopulationGenerator.py
    
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

class PopulationGenerator(object):
    """ Base class for generating C++ code from a population description. """
    def __init__(self, name, desc):
        self.class_name = name
        self.desc = desc
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.class_name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.class_name+'.pyx'
        
    def generate(self, verbose):
        self.verbose = verbose
        if verbose:
            if self.class_name != self.desc['name']:
                Global._print( 'Generating', self.desc['name'], '(', self.class_name, ')' ) 
            else:
                Global._print( 'Generating', self.class_name )

        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self.generate_header())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self.generate_body())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generate_pyx()) 
    
    
    def generate_members_declaration(self):
        """ Returns private members declaration. 
        
        Depend only on locality. rate should not be defined twice.
        """
        members = ""
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] == 'r': # the vector is already declared
                members += """
    // r_ : local
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
        """ Returns public members access. 
        
        Depend only on locality. 
        """
        members = ""
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in self.desc['local']: # local attribute
                members += local_variable_access % {'name' : param['name'], 
                                                    'type': param['ctype']}

            elif param['name'] in self.desc['global']: # global attribute
                members += global_variable_access % {'name' : param['name'], 
                                                     'type': param['ctype']}

        return members
    
    
    def generate_globalops_header(self):
        """ Returns access to global operations plus the private definition. 
        """
        access = ""
        method = ""
        for globop in self.desc['global_operations']:
            # Access method
            access += """
    // Global operation : %(function)s(%(variable)s)
    DATA_TYPE get%(Function)s%(Variable)s() { 
        return %(variable)s_%(function)s_; 
    }
""" % {'variable' : globop['variable'],
       'Variable' : globop['variable'].capitalize(),
       'function' : globop['function'],
       'Function' : globop['function'].capitalize()}
            # Internal definition + compute method
            method += """
    // Global operation : %(function)s(%(variable)s)
    void compute_%(function)s_%(variable)s();
    DATA_TYPE %(variable)s_%(function)s_;
""" % {'variable' : globop['variable'],
       'Variable' : globop['variable'].capitalize(),
       'function' : globop['function'],
       'Function' : globop['function'].capitalize()}
        return access, method
    
    def generate_random_definition(self):
        """ generate definition of random variables """
        definition = ""
        for var in self.desc['random_distributions']:
            definition += """
    std::vector<DATA_TYPE> %(name)s_;
    %(class)sDistribution%(template)s* %(dist)s_;
""" % {'name': var['name'], 
       'dist' : var['name'].replace('rand','dist'),
       'class': var['dist'] ,
       'template' : var['template']
      }
        return definition
    
    def generate_constructor(self):
        """ Content of the Population constructor and resetToInit."""
        constructor = ""
        reset=""
        # Attributes
        for param in self.desc['parameters'] + self.desc['variables']:            
            if param['name'] in self.desc['local']: # local attribute
                # Determine the initial value
                ctype = param['ctype']
                if ctype == 'bool':
                    cinit = 'true' if param['init'] else 'false'
                elif ctype == 'int':
                    cinit = int(param['init'])
                elif ctype == 'DATA_TYPE':
                    if isinstance(param['init'], np.ndarray): # will be set later
                        cinit = 0.0
                    else:
                        cinit = float(param['init'])
                    
                # Initialize the attribute
                constructor += """
    // %(name)s_ : local
    %(name)s_ = std::vector<%(type)s> (nbNeurons_, %(init)s);
    record_%(name)s_ = false; 
    recorded_%(name)s_ = std::vector< std::vector< %(type)s > >();    
""" % {'name' : param['name'], 'type': param['ctype'], 'init' : str(cinit)}
                reset += """
    // %(name)s_ : local
    %(name)s_ = std::vector<%(type)s> (nbNeurons_, %(init)s);   
""" % {'name' : param['name'], 'type': param['ctype'], 'init' : str(cinit)}

            elif param['name'] in self.desc['global']: # global attribute
                # Determine the initial value
                ctype = param['ctype']
                if ctype == 'bool':
                    cinit = 'true' if param['init'] else 'false'
                elif ctype == 'int':
                    cinit = int(param['init'])
                elif ctype == 'DATA_TYPE':
                    cinit = float(param['init'])
                # Initialize the attribute
                constructor += """
    // %(name)s_ : global
    %(name)s_ = %(init)s;   
""" % {'name' : param['name'], 'init': str(cinit)} 
                reset += """
    // %(name)s_ : global
    %(name)s_ = %(init)s;   
""" % {'name' : param['name'], 'init' : str(cinit)} 

        # Global operations
        for var in self.desc['global_operations']:
            constructor += """
    // Global operation : %(function)s(%(variable)s)
    %(variable)s_%(function)s_ = DATA_TYPE(0.0);
""" % {'variable' : var['variable'],
       'function' : var['function']}
        
        # Initialize dt
        constructor += """
    // dt : integration step
    dt_ = %(dt)s;
""" % { 'dt' : str(Global.config['dt'])}       
      
        # initialization of random distributions
        for var in self.desc['random_distributions']:
            constructor += """
    %(dist)s_ = new %(class)sDistribution%(template)s(%(args)s);
""" % { 'dist' : var['name'].replace('rand','dist'),
        'class': var['dist'],
        'args': var['args'] ,
        'template': var['template'] 
      }
        
        return constructor, reset
    
    def generate_destructor(self):
        """ Content of the Population destructor."""
        destructor = ""
        # Attributes
        for param in self.desc['parameters'] + self.desc['variables']:            
            if param['name'] in self.desc['local']: # local attribute
                destructor += """
    %(name)s_.clear();""" % {'name' : param['name']}
        return destructor
    
    def generate_globalops(self):
        """ Defines how global operations should be computed."""
        single = ""
        glob = ""
        for var in self.desc['global_operations']:
            # Content of global_operations()
            glob += """
    compute_%(function)s_%(variable)s();
""" % { 'function': var['function'] , 'variable': var['variable'] }
            # Single operation
            if var['function'] == 'min':
                single += """
void %(class)s::compute_min_%(var)s() {
    %(var)s_min_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        if(%(var)s_[i] < %(var)s_min_)
            %(var)s_min_ = %(var)s_[i];
    }
}
""" % { 'class': self.class_name, 'var': var['variable'] }
            elif var['function'] == 'max':
                single += """
void %(class)s::compute_max_%(var)s() {
    %(var)s_max_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        if(%(var)s_[i] > %(var)s_max_)
            %(var)s_max_ = %(var)s_[i];
    }
}
"""  % { 'class': self.class_name, 'var': var['variable'] }
            elif var['function'] == 'mean':
                single += """
void %(class)s::compute_mean_%(var)s() {
    %(var)s_mean_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        %(var)s_mean_ += %(var)s_[i];
    }
    
    %(var)s_mean_ /= %(var)s_.size();
}
"""  % { 'class': self.class_name, 'var': var['variable'] }
            elif var['function'] == 'sum':
                single += """
void %(class)s::compute_sum_%(var)s() {
    %(var)s_mean_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        %(var)s_mean_ += %(var)s_[i];
    }
}
"""  % { 'class': self.class_name, 'var': var['variable'] }
        return single, glob
    
    def generate_record(self):
        " Code for recording."
        code = ""
        # Attributes
        for param in self.desc['parameters'] + self.desc['variables']:            
            if param['name'] in self.desc['local']: # local attribute
                code += """
    if(record_%(var)s_)
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }
        return code
    

    def generate_global_metastep(self):
        """
        Code for the metastep.
        """
        code = ""
        for var in self.desc['random_distributions']:
            code +="""
    %(name)s_ = %(dist)s_->getValues(nbNeurons_);
""" % { 'name' : var['name'],
        'dist' : var['name'].replace('rand', 'dist') 
      }

        code += "\
        "
        for param in self.desc['variables']:            
            if param['name'] in self.desc['global']: # global attribute
                if '[i]' in param['cpp']:
                    Global._error('The global variable ' + param['cpp'] + \
                                  ' can not depend on local ones!')
                else:
                    code += """
%(cpp)s
""" % { 'cpp': param['cpp'] }
        return code
    
    def generate_cwrappers(self):
        "Parts of the C++ header which should be accessible to Cython"
        code = ""
        
        for param in self.desc['parameters'] + self.desc['variables']:
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_wrapper_pyx % {   'name': param['name'], 
                                                'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
        
                
            elif param['name'] in self.desc['global']: # global attribute
                code += global_wrapper_pyx % {  'name': param['name'], 
                                                'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
        return code
    
    def generate_pyfunctions(self):
        "Python functions accessing the Cython wrapper"
        code = ""        
        for param in self.desc['parameters'] + self.desc['variables']:
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_property_pyx % { 'name': param['name'], 
                                               'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
                
            elif param['name'] in self.desc['global']: # global attribute
                code += global_property_pyx % { 'name': param['name'], 
                                                'type': param['ctype'] if param['ctype'] != 'DATA_TYPE' else 'float'}
        return code
                
    
    def generate_functions(self):
        "Custom functions"
        code = ""        
        for func in self.desc['functions']:
            
            code += '    ' + func['cpp']
            
        return code