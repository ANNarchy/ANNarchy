""" 

    SpikeProjectionGenerator.py
    
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

from ProjectionGenerator import ProjectionGenerator
from Templates import *

class SpikeProjectionGenerator(ProjectionGenerator):
    """ Class for generating C++ code from a spike population description. """
    def __init__(self, name, desc):
        ProjectionGenerator.__init__(self, name, desc)
            
    def generate_header(self):
        " Generates the C++ header file."        
        # Access method for attributes
        access = self.generate_dendrite_access_declaration()
        
        # Custom function
        functions = self.generate_functions()
                
        # Generate the code
        template = spike_projection_header
        dictionary = { 
            'class': self.name, 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'functions': functions 
        }
        return template % dictionary

    def generate_body(self):
        
        # Generate code for the global variables
        global_learn = self.generate_globallearn()
        
        # Generate code for the local variables
        local_learn = self.generate_locallearn()
        
        # Access method for attributes
        access = self.generate_dendrite_access_definition()

        # Generate the code
        template = spike_projection_body
        dictionary = {         
            'class': self.name,
            'dend_class': self.name.replace('Projection', 'Dendrite'), 
            'add_include': self.generate_add_proj_include(),
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'access': access
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

        template = spike_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary    
    
        
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
