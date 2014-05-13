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
from ANNarchy.core.Random import *

from DendriteGenerator import DendriteGenerator
from Templates import *
from TemplatesOMP import *

class RateDendriteGenerator(DendriteGenerator):
    """ Class for generating C++ code from a rate dendrite description. """
    def __init__(self, name, desc):
        DendriteGenerator.__init__(self, name, desc)
            
    def generate_header(self):
        " Generates the C++ header file for dendrite class."        
        # Private members declarations
        members = self.generate_members_declaration()
        
        # Access method for attributes
        access = self.generate_members_access()
        
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
            psp_code = '(*pre_rates_)[rank_[i]] * value_[i];'
        # Generate the code
        template = psp_code_body
        dictionary = {
            'psp': psp_code, 
            'psp_const_delay': psp_code,
            'psp_dyn_delay' : psp_code.replace('(*pre_rates_)', 'delayedRates')
        }    
        return template % dictionary
    
    def generate_globallearn(self):
        " Generates code for the globalLearn() method for global variables."

        # Generate the code
        code = ""
        for param in self.desc['variables']:
            if param['name'] in self.desc['global']: # global attribute 
                # The code is already in 'cpp'
                code +="""
    %(comment)s
    %(code)s   
""" % { 'comment': '// ' + param['eq'],
        'code' : param['cpp']}
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
    if(%(var)s_ < %(val)s)
        %(var)s_ = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
                    if bound == 'max':
                        code += """
    if(%(var)s_ > %(val)s)
        %(var)s_ = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
        return code
    
    def generate_locallearn(self):
        " Generates code for the localLearn() method for local variables."

        # Generate the code
        local_learn = ""
        for param in self.desc['variables']:
            if param['name'] in self.desc['local']: # local attribute 
                # The code is already in 'cpp'
                local_learn +="""
    %(comment)s
    %(code)s   
""" % { 'comment': '// ' + param['eq'],
        'code' : param['cpp']}
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
                        local_learn += """
            if(%(var)s_[i] < %(val)s)
                %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
                    if bound == 'max':
                        local_learn += """
            if(%(var)s_[i] > %(val)s)
                %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : pval}
        
        if len(local_learn) > 1:
            #
            # build final code
            code="""
#ifdef _DEBUG
    std::cout << "Dendrite (n = " << post_neuron_rank_ << ", ptr = " << this << "): update " << nbSynapses_ << " synapse(s)." << std::endl;
#endif
    post_rate_ = (*post_rates_)[post_neuron_rank_];
    
    for(int i=0; i < nbSynapses_; i++) 
    {
 %(local_learn)s
    }
    """ % { 'local_learn': local_learn }
        else:
            code = ""
            
        return code