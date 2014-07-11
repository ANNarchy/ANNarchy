""" 

    RatePopulationGenerator.py
    
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
from ANNarchy.core.Random import RandomDistribution
from ANNarchy.generator.Utils import *
from PopulationGenerator import PopulationGenerator
from Templates import *


import re

class SpikePopulationGenerator(PopulationGenerator):
    """ Class for generating C++ code from a spike population description. """
    def __init__(self, name, desc):
        PopulationGenerator.__init__(self, name, desc)
        
        self.declared_targets = desc['targets']
        self.connected_targets = desc['pop'].targets
    
    def generate_members_declaration(self):
        members = PopulationGenerator.generate_members_declaration(self)

#         for variable in self.connected_targets: # Actually connected
#             members += """
#     // g_%(name)s_new_ : local
#     std::vector<DATA_TYPE> g_%(name)s_new_;
# """ % {'name' : variable}

        for variable in self.declared_targets: # Only declared
            if not 'g_'+variable in self.desc['local']:
                members += """
    // g_%(name)s_ : local
    std::vector<DATA_TYPE> g_%(name)s_;
""" % {'name' : variable}

        return members
    
    def generate_constructor(self):

        constructor= PopulationGenerator.generate_constructor(self)     

        for variable in self.declared_targets:
            if not 'g_'+variable in self.desc['local']:
                constructor += """
    // g_%(name)s_ : local
    g_%(name)s_ = std::vector<DATA_TYPE> (nbNeurons_, %(init)s);
""" % {'name' : variable,
       'init' : str(0.0)
       }
 
        return constructor
    
    def generate_header(self, template = None):
        " Generates the C++ header file."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
        # Global operations
        global_ops_access, global_ops_method = self.generate_globalops_header()
        
        # Random variables
        randoms = self.generate_random_definition()
        
        # Stop condition
        stop_condition = self.generate_stop_condition_definition()

        # Custom function
        functions = self.generate_functions()
        
        # connected projections
        friends = self.generate_friend_decl()
        
        # Generate the code
        if not template:
            template = spike_population_header
        dictionary = {
            'class' : self.class_name,
            'access' : access,
            'global_ops_access' : global_ops_access,
            'global_ops_method' : global_ops_method,
            'member' : members,
            'random' : randoms,
            'stop_condition' : stop_condition,
            'friend' : friends,
            'functions' : functions
        }
        return template % dictionary

    def generate_friend_decl(self):
        code = ""
        
        for proj in Global._projections:
            if (self.desc['name'] == proj.post.name) and \
               (proj.target in proj.post.targets):
                code+= """friend class %(name)s;""" % { 'name': proj.name.replace('Projection','Dendrite') }
                
        return code    
    
    def generate_body(self):
        " Generates the C++ .cpp file"
        # Constructor
        constructor = self.generate_constructor()

        # Destructor
        destructor = self.generate_destructor()

        # Single operations
        singleops, globalops = self.generate_globalops()

        # Record
        record = self.generate_record()

        # Stop condition
        stop_condition = self.generate_stop_condition_body()

        # Meta-step
        local_metastep = self.generate_local_metastep()
        global_metastep = self.generate_global_metastep()

        # reset event
        reset_event, refractory_event = self.generate_reset_event()
        
        # Generate the code
        template = spike_population_body
        dictionary = {
            'class' : self.class_name,
            'pop_id': self.desc['id'],
            'constructor' : constructor,
            'destructor' : destructor,
            'localMetaStep' : local_metastep,
            'globalMetaStep' : global_metastep,
            'global_ops' : globalops,
            'record' : record,
            'stop_condition' : stop_condition,
            'reset_event': reset_event,
            'refractory_event': refractory_event,
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
            'class_name': self.desc['class'],
            'name' :  self.desc['name'],
            'cFunction' : cwrappers, 
            'neuron_count' : size,
            'pyFunction' : pyfunctions,
        }
        return template % dictionary

    def generate_reset_event(self):
        desc = self.desc['spike']['spike_reset']
        reset_code = ''
        refractory_code = ''
        for var in desc:
            reset_code += ' '*8 + var['cpp'] + '\n'
            if not 'unless_refractory' in var['constraint']:
                refractory_code += ' '*8 + var['cpp'].replace('(*it)', 'rank') + '\n'
        
        return reset_code, refractory_code

    def generate_local_metastep(self):
        """
        Code for the metastep.
        """
        code = generate_equation_code(self.desc, 'local')

        # default = reset of target conductances
        for target in self.connected_targets:
            if not 'g_'+target in self.desc['local']:
                Global._warning('Using standard behavior for the conductance g_'+target + '.')
                code += """
    // Default behavior for g_%(target)s_
    g_%(target)s_[i] = 0.0;
""" % {'target' : target}

        # spike propagation and refractory period
        code += spike_emission_template % {'cond' : self.desc['spike']['spike_cond'] } 

        return code
