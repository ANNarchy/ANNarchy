""" 

    SpikeDendriteGenerator.py
    
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

from DendriteGenerator import DendriteGenerator
from Templates import *
from TemplatesOMP import *
import re

class SpikeDendriteGenerator(DendriteGenerator):
    """ Class for generating dendrite C++ code from a spike synapse description. """
    def __init__(self, name, desc, paradigm):
        DendriteGenerator.__init__(self, name, desc)
        
        self.paradigm = paradigm
            
    def generate_header(self):
        " Generates the C++ header file for dendrite class."        
        # Private members declarations
        members = self.generate_members_declaration()
         
        # Access method for attributes
        access = self.generate_members_access()
         
        # Custom function
        functions = self.generate_functions()
         
        # Generate the code
        template = spike_dendrite_header
        dictionary = { 
            'class': self.name.replace('Projection', 'Dendrite'), 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'member': members,
            'functions': functions 
        }
        return template % dictionary
     
    def generate_body(self):
        # Initialize parameters and variables
        constructor = self.generate_constructor()
         
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
         
        record = self.generate_record()
         
        # Generate the code
        template = spike_dendrite_body
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
 
        template = spike_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary    
     
    def generate_record(self):
        code = DendriteGenerator.generate_record(self)
 
        if 'pre_spike' in self.desc.keys():
            for param in self.desc['pre_spike']:
                if param['name'] in self.desc['local']:
                    continue

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
         
        dictionary = {
            'eq' : code,
            'target': self.desc['target']
        }
        return pre_event_body % dictionary
 
    def generate_post_event(self):
        """ """
        code = ""
 
        # generate additional statements        
        if 'post_spike' in self.desc.keys():
            for tmp in self.desc['post_spike']:
                code += """
    %(eq)s
""" % { 'eq' : tmp['eq'] }
         
        dictionary = {
            'eq' : code,
            'target': self.desc['target']
        }
        return post_event_body % dictionary
     
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
