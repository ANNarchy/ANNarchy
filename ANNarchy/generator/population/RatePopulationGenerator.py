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
from ANNarchy.core.Random import *

from PopulationGenerator import PopulationGenerator
from Templates import *

import re

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
        
        # Custom function
        functions = self.generate_functions()
                
        # Generate the code
        template = rate_population_header
        dictionary = {
            'class' : self.class_name,
            'access' : access,
            'global_ops_access' : global_ops_access,
            'global_ops_method' : global_ops_method,
            'member' : members,
            'random' : randoms,
            'functions' : functions
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
            'class' : self.class_name,
            'pop_id': self.desc['id'],
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
            'class_name': self.desc['class'],
            'name' : self.desc['name'],
            'cFunction' : cwrappers, 
            'neuron_count' : size,
            'pyFunction' : pyfunctions,
        }
        return template % dictionary

    def generate_prepare_neurons(self):
        return rate_prepare_neurons

    def generate_local_metastep(self):
        """
        Code for the metastep.
        """
        code = ""         
        for param in self.desc['variables']: 
            # Local attributes           
            if param['name'] in self.desc['local']: 
                code += """
    %(comment)s
    %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }

            # Process the bounds min and max
            for bound, val in param['bounds'].iteritems():
                # Bound min
                if bound == 'min':
                    code += """
    if(%(var)s_[i] < %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}
                # Bound max 
                if bound == 'max':
                    code += """
    if(%(var)s_[i] > %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}
        return code