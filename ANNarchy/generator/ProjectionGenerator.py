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
from ANNarchy.core.Random import *

from ANNarchy.generator import ProjectionTemplates as Templates

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
        
        local_template = Templates.local_variable_access
        global_template = Templates.global_variable_access
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "value":
                func = """
    std::vector< std::vector< %(type)s > >getRecorded%(Name)s() { return this->recorded_%(name)s_; }                    
    void startRecord%(Name)s() { this->record_%(name)s_ = true; }
    void stopRecord%(Name)s() { this->record_%(name)s_ = false; }
    void clearRecorded%(Name)s() { this->recorded_%(name)s_.clear(); }
"""
                members += func % { 'name' : param['name'], 
                                    'Name': param['name'].capitalize(),
                                    'type': param['ctype']}
            elif param['name'] in self.desc['local']: # local attribute
                members += local_template % { 'name' : param['name'], 
                                              'Name': param['name'].capitalize(),
                                              'type': param['ctype']}
                
            elif param['name'] in self.desc['global']: # global attribute
                members += global_template % { 'name' : param['name'], 
                                               'Name': param['name'].capitalize(),
                                               'type': param['ctype']}

        return members

    def generate_dendrite_access_deklaration(self):
        """ 
        Returns access methods towards the attached dendrites 
        
        Notes:
            * depend only on locality.
            * rank, delay access is already implemented in base class
            * values are partly predefined
            * the definition of the function is implemented in the body 
        """
        members_header = ""
        
        local_template = Templates.local_idx_variable_access
        global_template = Templates.global_idx_variable_access
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "value":
                func = """
    std::vector< std::vector< %(type)s > >getRecorded%(Name)s(int post_rank);                    
    void startRecord%(Name)s(int post_rank);
    void stopRecord%(Name)s(int post_rank);
    void clearRecorded%(Name)s(int post_rank);
"""             
                members_header += func % { 'name' : param['name'], 
                                           'Name': param['name'].capitalize(),
                                           'type': param['ctype'],
                                           'class': self.name.replace('Projection', 'Dendrite')}
            elif param['name'] in self.desc['local']: # local attribute
                members_header += local_template % { 'name' : param['name'], 
                                              'Name': param['name'].capitalize(),
                                              'type': param['ctype'],
                                              'class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['global']: # global attribute
                members_header += global_template % { 'name' : param['name'], 
                                               'Name': param['name'].capitalize(),
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
        
        local_template = Templates.local_idx_variable_access_body
        global_template = Templates.global_idx_variable_access_body
        
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in ['rank', 'delay', 'psp']: # Already declared
                continue
            
            if param['name'] == "value":
                func = """
std::vector< std::vector< %(type)s > > %(class)s::getRecorded%(Name)s(int post_rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getRecorded%(Name)s(); 
}                    
void %(class)s::startRecord%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->startRecord%(Name)s(); 
}
void %(class)s::stopRecord%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->stopRecord%(Name)s(); 
}
void %(class)s::clearRecorded%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->clearRecorded%(Name)s(); 
}
"""             
                members_body += func % { 'name' : param['name'], 
                                         'Name': param['name'].capitalize(),
                                         'type': param['ctype'],
                                         'class': self.name,
                                         'dend_class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['local']: # local attribute
                members_body += local_template % { 'name' : param['name'], 
                                              'Name': param['name'].capitalize(),
                                              'type': param['ctype'],
                                              'class': self.name,
                                              'dend_class': self.name.replace('Projection', 'Dendrite')}
                
            elif param['name'] in self.desc['global']: # global attribute
                members_body += global_template % { 'name' : param['name'], 
                                               'Name': param['name'].capitalize(),
                                               'type': param['ctype'],
                                               'class': self.name,
                                               'dend_class': self.name.replace('Projection', 'Dendrite')}

        return members_body
    
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
        for param in self.desc['parameters'] + self.desc['variables']:
            if param['name'] in self.desc['local']: # local attribute
                code += """
    if(record_%(var)s_)
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }

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
    

class RateProjectionGenerator(ProjectionGenerator):
    """ Class for generating C++ code from a rate population description. """
    def __init__(self, name, desc):
        ProjectionGenerator.__init__(self, name, desc)
            
    def generate_header(self):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access_header = self.generate_dendrite_access_deklaration()
        
        # Custom function
        functions = self.generate_functions()
        
        # Generate the code
        template = Templates.rate_projection_header
        dictionary = { 
            'class': self.name, 
            'dend_class': self.name.replace('Projection', 'Dendrite'),
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access_header,
            'member': members,
            'functions': functions }
        return template % dictionary

    def generate_body(self):
        # Initialize parameters and variables
        constructor = self.generate_constructor()
    
        # Access method for attributes
        access_body = self.generate_dendrite_access_definition()
        
        # Generate the code
        template = Templates.rate_projection_body
        dictionary = {         
            'class': self.name,
            'add_include': self.generate_add_proj_include(),
            'dend_class': self.name.replace('Projection', 'Dendrite'), 
            'access': access_body,
            'destructor': self.generate_destructor() ,
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'init': constructor
        } 
        return template % dictionary
    
    def generate_pyx(self):
        """
        Generate complete cython wrapper class
            
        Notes:
            * dependent on coding.
        """
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code

        template = Templates.rate_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary    
    
class RateProjectionGeneratorCUDA(ProjectionGenerator):
    """ Class for generating CUDA/C++ code from a rate population description. """
    def __init__(self, name, desc):
        ProjectionGenerator.__init__(self, name, desc)
            
    def generate_header(self):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
        # Custom function
        functions = self.generate_functions()
        
        # Generate the code
        template = Templates.rate_projection_header
        dictionary = { 
            'class': self.name, 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'member': members,
            'functions': functions }
        return template % dictionary
    
    def generate_body(self):
        # Initialize parameters and variables
        constructor = self.generate_constructor()
        
        # Computation of psp for the weighted sum
        psp = self.generate_psp()
        
        # Generate code for the global variables
        global_learn = self.generate_globallearn()
        
        # Generate code for the local variables
        local_learn = self.generate_locallearn()
        
        # structural plasticity
        add_synapse = self.generate_add_synapse()
        rem_synapse = self.generate_rem_synapse()
        rem_all_synapse = self.generate_rem_all_synapse()

        record = ""
        
        # Generate the code
        template = Templates.rate_projection_body
        dictionary = {         
            'class': self.name,
            'add_include': self.generate_add_proj_include(),
            'destructor': '' ,
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'init': constructor, 
            'sum': psp, 
            'local': local_learn, 
            'global': global_learn,
            'record' : record,
            'add_synapse_body': add_synapse,
            'rem_synapse_body': rem_synapse,
            'rem_all_synapse_body': rem_all_synapse }
        return template % dictionary

    def generate_pyx(self):
        """
        Generate complete cython wrapper class
            
        Notes:
            * dependent on coding.
        """
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code

        template = Templates.rate_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary
        
class SpikeProjectionGenerator(ProjectionGenerator):
    """ Class for generating C++ code from a spike population description. """
    def __init__(self, name, desc):
        ProjectionGenerator.__init__(self, name, desc)
            
    def generate_proj_header(self):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_idx_members_access_header()
        
        # Custom function
        functions = self.generate_functions()
                
        # Generate the code
        template = Templates.spike_projection_header
        dictionary = { 
            'class': self.name, 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'member': members,
            'functions': functions 
        }
        return template % dictionary

    def generate_proj_body(self):
        
        # Computation of psp for the weighted sum
        psp = self.generate_psp()
        
        # Generate code for the global variables
        global_learn = self.generate_globallearn()
        
        # Generate code for the local variables
        local_learn = self.generate_locallearn()
        
        # Access method for attributes
        access = self.generate_idx_members_access_body()

        # structural plasticity
        add_synapse = self.generate_add_synapse()
        rem_synapse = self.generate_rem_synapse()
        rem_all_synapse = self.generate_rem_all_synapse()
        
        record = self.generate_record()

        # Generate the code
        template = Templates.spike_projection_body
        dictionary = {         
            'class': self.name,
            'dend_class': self.name.replace('Projection', 'Dendrite'), 
            'add_include': self.generate_add_proj_include(),
            'destructor': self.generate_destructor() ,
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'init': self.generate_constructor(), 
            'access': access,
            'local': local_learn, 
            'global': global_learn,
            'add_synapse_body': add_synapse,
            'rem_synapse_body': rem_synapse,
            'rem_all_synapse_body': rem_all_synapse,
            'record' : record 
        }
        return template % dictionary
    
    def generate_dendrite_header(self):
        " Generates the C++ header file for dendrite class."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
        # Custom function
        functions = self.generate_functions()
        
        # Generate the code
        template = DendTemplates.spike_dendrite_header
        dictionary = { 
            'class': self.name.replace('Projection', 'Dendrite'), 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'member': members,
            'functions': functions 
        }
        return template % dictionary
    
    def generate_dendrite_body(self):
        # Initialize parameters and variables
        constructor = self.generate_constructor()
        
        # Computation of psp for the weighted sum
        psp = self.generate_psp()
        
        # Generate code for the global variables
        global_learn = self.generate_globallearn()
        
        # Generate code for the local variables
        local_learn = self.generate_locallearn()
        
        # Generate code for the pre- and postsynaptic events
        pre_event = self.generate_pre_event()
        post_event = self.generate_post_event()
        
        # structural plasticity
        add_synapse = self.generate_add_synapse()
        rem_synapse = self.generate_rem_synapse()
        rem_all_synapse = self.generate_rem_all_synapse()
        
        record = ""
        
        # Generate the code
        template = DendTemplates.spike_dendrite_body
        dictionary = {         
            'class': self.name.replace('Projection','Dendrite'),
            'add_include': self.generate_add_proj_include(),
            'destructor': self.generate_destructor(),
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'init': constructor, 
            'local': local_learn, 
            'global': global_learn,#
            'pre_event': pre_event,
            'post_event': post_event,
            'record' : record,
            'add_synapse_body': add_synapse,
            'rem_synapse_body': rem_synapse,
            'rem_all_synapse_body': rem_all_synapse 
        }
        return template % dictionary
        
    def generate_pyx(self):
        """
        Generate complete cython wrapper class
            
        Notes:
            * dependent on coding and paradigm
        """
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code

        template = Templates.spike_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary    
    
    def generate_record(self):
        code = ProjectionGenerator.generate_record(self)

        if 'pre_spike' in self.desc.keys():
            for param in self.desc['pre_spike']:
                if param['name'] in self.desc['local']:
                    continue
                
                #
                # if a variable matches g_exc, g_inh ... we skip
                if re.findall("(?<=g\_)[A-Za-z]+", param['name']) != []:
                    #print 'skipped', param['name']
                    continue
                
                code += """
    if ( record_%(var)s_ )
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }

        if 'post_spike' in self.desc.keys():
            for param in self.desc['post_spike']:
                if param['name'] in self.desc['local']:
                    continue
                
                code += """
    if ( record_%(var)s_ )
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }

        return code
        
    def generate_pre_event(self):
        """ """
        code = ""

        # generate additional statements        
        if 'pre_spike' in self.desc.keys():
            for tmp in self.desc['pre_spike']:
                code += """
    %(eq)s
""" % { 'eq' : tmp['eq'] }
        
        template = OMPTemplates.pre_event_body
        dictionary = {
            'eq' : code,
            'target': self.desc['target']
        }
        return template % dictionary

    def generate_post_event(self):
        """ """
        code = ""

        # generate additional statements        
        if 'post_spike' in self.desc.keys():
            for tmp in self.desc['post_spike']:
                code += """
    %(eq)s
""" % { 'eq' : tmp['eq'] }
        
        template = OMPTemplates.post_event_body
        dictionary = {
            'eq' : code,
            'target': self.desc['target']
        }
        return template % dictionary
     
    def generate_psp(self):
        " Generates code for the computeSum() method depending on psp variable of the synapse."
        return ""
    
    def generate_globallearn(self):
        return ""
    
    def generate_locallearn(self):
        code = """
    for(int i=0; i<(int)rank_.size();i++) 
    {
"""
        for param in self.desc['variables']:
            if param['name'] in self.desc['local']: # local attribute 
                # The code is already in 'cpp'
                code +="""
        %(code)s   
""" % {'code' : param['cpp']}
                # Set the min and max values 
                for bound, val in param['bounds'].iteritems():
                    # Check if the min/max value is a float/int or another parameter/variable
                    if val in self.desc['local']:
                        pval = val + '_[i]'
                    elif val in self.desc['global']:
                        pval = val + '_'
                    else:
                        pval = val
                    if bound == 'min':
                        code += """
        if(%(var)s_[i] < %(val)s)
            %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
                    if bound == 'max':
                        code += """
        if(%(var)s_[i] > %(val)s)
            %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
        code+="""
    }
"""
        return code
