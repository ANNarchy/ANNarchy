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
from ANNarchy4.core import Global
from ANNarchy4.generator.PopulationTemplates import *
from ANNarchy4.core.Random import *

import re

class PopulationGenerator(object):
    """ Base class for generating C++ code from a population description. """
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.name+'.pyx'
        
    def generate(self, verbose):
        self.verbose = verbose
        if verbose:
            if self.name != self.desc['name']:
                Global._print( 'Generating', self.desc['name'], '(', self.name, ')' ) 
            else:
                Global._print( 'Generating', self.name )

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
            if param['name'] == 'rate': # the vector is already declared
                members += """
    // rate_ : local
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
                                                    'Name': param['name'].capitalize(),
                                                    'type': param['ctype']}

            elif param['name'] in self.desc['global']: # global attribute
                members += global_variable_access % {'name' : param['name'], 
                                                     'Name': param['name'].capitalize(),
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
        definition = """
    // Random variables"""
        for var in self.desc['random_distributions']:
            definition += """
    std::vector<DATA_TYPE> %(name)s_;
""" % {'name': var['name']}
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
""" % { 'class': self.name, 'var': var['variable'] }
            elif var['function'] == 'max':
                single += """
void %(class)s::compute_max_%(var)s() {
    %(var)s_max_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        if(%(var)s_[i] > %(var)s_max_)
            %(var)s_max_ = %(var)s_[i];
    }
}
"""  % { 'class': self.name, 'var': var['variable'] }
            elif var['function'] == 'mean':
                single += """
void %(class)s::compute_mean_%(var)s() {
    %(var)s_mean_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        %(var)s_mean_ += %(var)s_[i];
    }
    
    %(var)s_mean_ /= %(var)s_.size();
}
"""  % { 'class': self.name, 'var': var['variable'] }
            elif var['function'] == 'sum':
                single += """
void %(class)s::compute_sum_%(var)s() {
    %(var)s_mean_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        %(var)s_mean_ += %(var)s_[i];
    }
}
"""  % { 'class': self.name, 'var': var['variable'] }
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
            try:
                dist = eval(var['definition'])
            except:
                Global._error('Random distribution ' + var['definition'] + ' is not valid.')
                exit(0)
            code +="""
    %(name)s_ = %(call)s.getValues(nbNeurons_);
""" % {'name' : var['name'], 'call' : dist._gen_cpp() }

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
    
    def generate_local_metastep(self):
        """
        Code for the metastep.
        """
        code = ""         
        for param in self.desc['variables']:            
            if param['name'] in self.desc['local']: # local attribute
                code += """
    %(cpp)s
""" % { 'cpp': param['cpp'] }

            if self.desc.has_key('spike'):
                if param['name'] == self.desc['spike']['name']:
                    code += """
    if( %(cond)s )
    {
        #pragma omp critical
        {
            //std::cout << "emit spike (pop " << name_ <<")["<<i<<"] ( time="<< ANNarchy_Global::time<< ")" << std::endl;
            this->propagate_.push_back(i);
            this->reset_.push_back(i);
            
            spike_timings_[i].push_back(ANNarchy_Global::time);
            spiked_[i] = true;
        }
    }
""" % {'cond' : self.desc['spike']['spike_cond'] } #TODO: check code

            # Process the bounds min and max
            for bound, val in param['bounds'].iteritems():
                # Check if the min/max value is a float/int or another parameter/variable
                if val in self.desc['local']:
                    pval = val + '_[i]'
                elif val in self.desc['global']:
                    pval = val + '_'
                else:
                    pval = val
                # Bound min
                if bound == 'min':
                    code += """
    if(%(var)s_[i] < %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
                # Bound max 
                if bound == 'max':
                    code += """
    if(%(var)s_[i] > %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
        return code
    
    def generate_cwrappers(self):
        "Parts of the C++ header which should be accessible to Cython"
        code = ""
        
        for param in self.desc['parameters'] + self.desc['variables']:
            
            if param['name'] in self.desc['local']: # local attribute
                tmp_code = local_wrapper_pyx % { 'Name': param['name'].capitalize(), 
                                                 'name': param['name'], 
                                                 'type': param['ctype'] }
        
                code += tmp_code.replace('DATA_TYPE', 'float') # no double in cython
                
            elif param['name'] in self.desc['global']: # global attribute
                tmp_code = global_wrapper_pyx % { 'Name': param['name'].capitalize(), 
                                                  'name': param['name'], 
                                                  'type': param['ctype'] }
        
                code += tmp_code.replace('DATA_TYPE', 'float')
        return code
    
    def generate_pyfunctions(self):
        "Python functions acessing the Cython wrapper"
        code = ""        
        for param in self.desc['parameters'] + self.desc['variables']:
            
            if param['name'] in self.desc['local']: # local attribute
                code += local_property_pyx % { 'Name': param['name'].capitalize(), 
                                               'name': param['name'] }
                
            elif param['name'] in self.desc['global']: # global attribute
                code += global_property_pyx % { 'Name': param['name'].capitalize(), 
                                                'name': param['name'] }
        return code
                
    

class RatePopulationGenerator(PopulationGenerator):
    """ Class for generating C++ code from a rate population description. """
    
    def __init__(self, name, desc):
        PopulationGenerator.__init__(self, name, desc)
        
    def generate_header(self):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
        # Global operations
        global_ops_access, global_ops_method = self.generate_globalops_header()
        
        # Random variables
        randoms = self.generate_random_definition()
                
        # Generate the code
        template = rate_population_header
        dictionary = {
            'class' : self.name,
            'access' : access,
            'global_ops_access' : global_ops_access,
            'global_ops_method' : global_ops_method,
            'member' : members,
            'random' : randoms
        }
        return template % dictionary
    
    def generate_body(self):
        " Generates the C++ .cpp file"
        # Constructor
        constructor, reset = self.generate_constructor()
        # Destructor
        destructor = self.generate_destructor()
        # prepare neurons (shift delayed rates)
        prepare = self.generate_prepare_neurons()
        # Single operations
        singleops, globalops = self.generate_globalops()
        # Record
        record = self.generate_record()
        # Meta-step
        local_metastep = self.generate_local_metastep()
        global_metastep = self.generate_global_metastep()
        
        # Generate the code
        template = rate_population_body
        dictionary = {
            'class' : self.name,
            'constructor' : constructor,
            'destructor' : destructor,
            'prepare': prepare,
            'resetToInit' : reset,
            'localMetaStep' : local_metastep,
            'globalMetaStep' : global_metastep,
            'global_ops' : globalops,
            'record' : record,
            'single_global_ops' : singleops
        }
        return template % dictionary
    
    def generate_pyx(self):
        # Get the size of the population
        size  = Global.get_population(self.desc['name']).size
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code
        template = rate_population_pyx
        dictionary = {
            'name' : self.name,
            'cFunction' : cwrappers, 
            'neuron_count' : size,
            'pyFunction' : pyfunctions,
        }
        return template % dictionary

    def generate_prepare_neurons(self):
        return rate_prepare_neurons

class SpikePopulationGenerator(PopulationGenerator):
    """ Class for generating C++ code from a spike population description. """
    def __init__(self, name, desc):
        PopulationGenerator.__init__(self, name, desc)
        
        self._app_proj = []
        for proj in Global._projections:
            if (self.desc['name'] == proj.post.name) and \
               (proj.target in proj.post.targets):
                self._app_proj.append(proj)
    
    def generate_members_declaration(self):
        members = PopulationGenerator.generate_members_declaration(self)

        if self._app_proj:
            for variable in self.desc['variables']:
                if re.findall("(?<=g_)[A-Za-z]+", variable['name']):
                     members += """
    // %(name)s_ : local
    std::vector<%(type)s> %(name)s_;
""" % {'name' : variable['name']+'_new', 'type': variable['ctype']}

        return members
    
    def generate_constructor(self):
        constructor, reset = PopulationGenerator.generate_constructor(self)

        if self._app_proj:
            for variable in self.desc['variables']:
                if re.findall("(?<=g_)[A-Za-z]+", variable['name']):
                    constructor += """
    %(name)s_new_ = std::vector<%(type)s> (nbNeurons_, %(init)s);
""" % {'name' : variable['name'], 
       'type': variable['ctype'], 
       'init' : str(0.0)
       }
                    reset += """
    %(name)s_new_ = std::vector<%(type)s> (nbNeurons_, %(init)s);
""" % {'name' : variable['name'], 
       'type': variable['ctype'], 
       'init' : str(0.0)
       }       
 
        return constructor, reset
    
    def generate_prepare_neurons(self):
        prepare = ""
        if self._app_proj:
            for variable in self.desc['variables']:
                if re.findall("(?<=g_)[A-Za-z]+", variable['name']):
                     prepare += """
    
    // add the new conductance to the old one
    // and reset the new conductance values
    std::transform(%(name)s_.begin(), %(name)s_.end(),
                   %(name)s_new_.begin(), %(name)s_.begin(),
                   std::plus<%(type)s>());
    std::fill(%(name)s_new_.begin(), %(name)s_new_.end(), 0.0);
    
""" % {'name' : variable['name'], 'type': variable['ctype']}
        
        return prepare
        
    def generate_header(self):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
        # Global operations
        global_ops_access, global_ops_method = self.generate_globalops_header()
        
        # Random variables
        randoms = self.generate_random_definition()
        
        # connected projections
        friends = self.generate_friend_decl()
        # Generate the code
        template = spike_population_header
        dictionary = {
            'class' : self.name,
            'access' : access,
            'global_ops_access' : global_ops_access,
            'global_ops_method' : global_ops_method,
            'member' : members,
            'random' : randoms,
            'friend' : friends
        }
        return template % dictionary

    def generate_friend_decl(self):
        code = ""
        
        for proj in Global._projections:
            if (self.desc['name'] == proj.post.name) and \
               (proj.target in proj.post.targets):
                code+= """friend class %(name)s;""" % { 'name': proj.name }
                
        return code    
    
    def generate_body(self):
        " Generates the C++ .cpp file"
        # Constructor
        constructor, reset = self.generate_constructor()
        # Destructor
        destructor = self.generate_destructor()
        # prepare neurons
        prepare = self.generate_prepare_neurons()
        # Single operations
        singleops, globalops = self.generate_globalops()
        # Record
        record = self.generate_record()
        # Meta-step
        local_metastep = self.generate_local_metastep()
        global_metastep = self.generate_global_metastep()

        # reset event
        reset_event = self.generate_reset_event()
        
        # Generate the code
        template = spike_population_body
        dictionary = {
            'class' : self.name,
            'constructor' : constructor,
            'destructor' : destructor,
            'prepare': prepare,
            'resetToInit' : reset,
            'localMetaStep' : local_metastep,
            'globalMetaStep' : global_metastep,
            'global_ops' : globalops,
            'record' : record,
            'reset_event': reset_event,
            'single_global_ops' : singleops
        }
        return template % dictionary
    
    def generate_pyx(self):
        # Get the size of the population
        size  = Global.get_population(self.desc['name']).size
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code
        template = spike_population_pyx
        dictionary = {
            'name' : self.name,
            'cFunction' : cwrappers, 
            'neuron_count' : size,
            'pyFunction' : pyfunctions,
        }
        return template % dictionary

    def generate_reset_event(self):
        code = self.desc['spike']['spike_reset'].replace('[i]','[(*it)]')
        
        return code
