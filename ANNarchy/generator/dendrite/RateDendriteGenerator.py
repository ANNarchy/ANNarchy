""" 

    RateDendriteGenerator.py
    
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
from TemplatesCUDA import *
from TemplatesRateCode import *

class RateDendriteGenerator(DendriteGenerator):
    """ Class for generating C++ code from a rate dendrite description. """
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
        template = rate_dendrite_header
        dictionary = { 
            'class': self.name.replace('Projection', 'Dendrite'), 
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access,
            'member': members,
            'random': random,
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
        
        record = self.generate_record()
        
        # Generate the code
        template = rate_dendrite_body
        dictionary = {         
            'class': self.name.replace('Projection','Dendrite'),
            'add_include': self.generate_add_proj_include(),
            'constructor': constructor,
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

    def generate_psp(self):
        " Generates code for the computeSum() method depending on psp variable of the synapse."
        # Get the psp information
        if 'psp' in self.desc.keys():
            psp_code = self.desc['psp']['cpp']
        else:
            psp_code = '(*pre_rates_)[rank_[i]] * w_[i];'

        # Generate the code
        dictionary = {
            'psp_no_delay': psp_code, 
            'psp_const_delay': psp_code,
            'psp_dyn_delay' : psp_code.replace('(*pre_rates_)', 'delayedRates')
        }    

        # Select the template according delay and paradigm
        #
        # Assumption: all dendrites have similar structure
        if self.desc['csr'] and self.desc['csr'].get_max_delay() > 1:
            if self.desc['csr'].uniform_delay():        
                template = psp_code_const_delay_omp if (self.paradigm == "openmp") else psp_code_body_cuda
            else:
                template = psp_code_dyn_delay_omp if (self.paradigm == "openmp") else psp_code_body_cuda    
        else:
            template = psp_code_no_delay_omp if (self.paradigm == "openmp") else psp_code_body_cuda

        return template % dictionary
    
    def generate_globallearn(self):
        """ Generates code for the globalLearn() method for global variables. """

        # Generate the code
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
        " Generates code for the localLearn() method for local variables."
        nb_var = 0
        for param in self.desc['variables']: 
            if param['name'] in self.desc['local']: 
                nb_var +=1
        if nb_var == 0:
            return ''
        
        code="""
#ifdef _DEBUG
    std::cout << "Dendrite (n = " << post_neuron_rank_ << ", ptr = " << this << "): update " << nbSynapses_ << " synapse(s)." << std::endl;
#endif
    post_r_ = (*post_rates_)[post_neuron_rank_];
    
    for(int i=0; i < nbSynapses_; i++) 
    {
 %(local_learn)s
    }
    """ % { 'local_learn': generate_equation_code(self.desc, 'local') }
            
        return code