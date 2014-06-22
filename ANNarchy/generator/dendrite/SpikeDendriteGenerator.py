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
from ANNarchy.generator.Utils import *

from DendriteGenerator import DendriteGenerator
from Templates import *
from TemplatesOMP import *
from TemplatesSpiking import *

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
        
        # random definition
        random = self.generate_random_definition()

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
            'random': random,
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
        pre_event_psp = self.generate_pre_event_psp()
        pre_event_learn = self.generate_pre_event_learn()
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
            'constructor': constructor,
            'destructor': self.generate_destructor(),
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class'],
            'init': constructor, 
            'local': local_learn, 
            'global': global_learn,
            'pre_event_psp': pre_event_psp,
            'pre_event_learn': pre_event_learn,
            'lside': pre_event_psp.split("+=")[0],
            'rside': pre_event_psp.split("+=")[1],
            'post_event': post_event,
            'record' : record,
            'add_synapse_body': add_synapse,
            'rem_synapse_body': rem_synapse,
            'rem_all_synapse_body': rem_all_synapse 
        }
        return template % dictionary  
     
    def generate_record(self):
        code = DendriteGenerator.generate_record(self)
 
        added = [] # list of already added variables, to prevent double adding, e. g. w
                
        if 'pre_spike' in self.desc.keys():
            for param in self.desc['pre_spike']:
                if param['name'] in self.desc['local']:
                    continue

                # if a variable matches g_exc, g_inh ... we skip
                if re.findall("(?<=g\_)[A-Za-z]+", param['name']) != []:
                    #print 'skipped', param['name']
                    continue
                
                added += param['name']
                code += """
    if ( record_%(var)s_ )
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }
 
        if 'post_spike' in self.desc.keys():
            for param in self.desc['post_spike']:
                if param['name'] in self.desc['local'] + added:
                    continue
                 
                code += """
    if ( record_%(var)s_ )
        recorded_%(var)s_.push_back(%(var)s_);
""" % { 'var': param['name'] }
 
        return code

    def generate_pre_event_psp(self):
        """ """
        code = ""
        if 'pre_spike' in self.desc.keys():
            for tmp in self.desc['pre_spike']:
                if tmp['name'] == "g_"+self.desc['target']:
                    code += """ 
    %(eq)s
""" % { 'eq' : tmp['eq'] }
         
        dictionary = {
            'eq' : code,
            'target': self.desc['target']
        }
        return conductance_body % dictionary

    def generate_pre_event_learn(self):
        """ """
        code = ""
 
        # generate additional statements        
        if 'pre_spike' in self.desc.keys():
            for tmp in self.desc['pre_spike']:
                if tmp['name'] == "g_"+self.desc['target']:
                    continue
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
        """ Generates code for the globalLearn() method for global variables. """
        code = ""
        for var in self.desc['random_distributions']:
            code +="""
    %(name)s_ = %(dist)s_->getValues(nbSynapses_);
""" % { 'name' : var['name'],
        'dist' : var['name'].replace('rand', 'dist') 
      }

        code += generate_equation_code(self.desc, 'global')
        
        return code
     
    def generate_locallearn(self):
        """ Generates code for the localLearn() method for local variables. """
        nb_var = 0
        for param in self.desc['variables']: 
            if param['name'] in self.desc['local']: 
                nb_var +=1
        if nb_var == 0:
            return ''
            
        code = """
    for(int i=0; i<(int)rank_.size();i++) 
    {
%(code)s
    }
""" % {'code': generate_equation_code(self.desc, 'local')}
        return code
