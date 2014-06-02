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

class SpikePopulationGenerator(PopulationGenerator):
    """ Class for generating C++ code from a spike population description. """
    def __init__(self, name, desc):
        PopulationGenerator.__init__(self, name, desc)
        
        self._app_proj = []
        for proj in Global._projections:
            if (self.desc['name'] == proj.post.name) and \
               (proj.target in proj.post.targets):
                self._app_proj.append(proj)
        print self._app_proj
    
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
        
        # Custom function
        functions = self.generate_functions()
        
        # connected projections
        friends = self.generate_friend_decl()
        # Generate the code
        template = spike_population_header
        dictionary = {
            'class' : self.class_name,
            'access' : access,
            'global_ops_access' : global_ops_access,
            'global_ops_method' : global_ops_method,
            'member' : members,
            'random' : randoms,
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
            'reset_event': reset_event,
            'reset_neuron': reset_event.replace('(*it)', 'rank'),
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
        code = self.desc['spike']['spike_reset'].replace('[i]','[(*it)]')
        
        return code

    def generate_local_metastep(self):
        """
        Code for the metastep.
        """
        code = ""         
        for param in self.desc['variables']:  
             # local attribute          
            if param['name'] in self.desc['local']:
                code += """
    %(comment)s
    %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }

            # spike propagation and refractory period
            if param['name'] == self.desc['spike']['name']:
                code += """
    if( %(cond)s )
    {
        if (refractory_counter_[i] < 1)
        {
            #pragma omp critical
            {
                //std::cout << "emit spike (pop " << name_ <<")["<<i<<"] ( time="<< ANNarchy_Global::time<< ")" << std::endl;
                this->propagate_.push_back(i);
                this->reset_.push_back(i);
                
                lastSpike_[i] = ANNarchy_Global::time;
                if(record_spike_){
                    spike_timings_[i].push_back(ANNarchy_Global::time);
                }
                spiked_[i] = true;
            }
        }
    }
""" % {'cond' : self.desc['spike']['spike_cond'] } #TODO: check code

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


        # default = reset of target conductances

        return code
