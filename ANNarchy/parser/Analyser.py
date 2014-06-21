"""

    Analyser.py

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
from ANNarchy.core.Neuron import RateNeuron
from ANNarchy.core.Synapse import RateSynapse
from ANNarchy.core.Global import _error, _warning, authorized_keywords, config
from ANNarchy.parser.Equation import Equation
from ANNarchy.parser.Function import FunctionParser
from ANNarchy.parser.StringManipulation import *
from ANNarchy.parser.ITE import *
from ANNarchy.parser.Extraction import *

from pprint import pprint
import re


class Analyser(object):
    """ Main class which analyses the network structure and equations in order to generate the C++ files."""

    def __init__(self, populations, projections):
        """ Constructor, called with Global._populations and Global._projections by default."""
    
        self.populations = populations
        self.projections = projections
        
        self.analysed_populations = {}
        self.analysed_projections = {}
        
    def analyse(self):
        """ Extracts all the relevant information in the network to prepare code generation."""
                       
        # Generate C++ code for all population variables 
        for pop in self.populations:      
            
            # Make sure population have targets declared only once 
            pop.targets = list(set(pop.targets))  
            pop.sources = list(set(pop.sources))

            for t in  pop.description['targets']:
                if not t in pop.targets:
                    _warning('The target ' + t + ' is used in the neuron of population ' + pop.name + ' but not connected.')
            
            # internal id
            pop.description['id'] = pop._id
            
            # Actualize initial values
            for variable in pop.description['parameters']:
                if isinstance(pop.init[variable['name']], bool) or \
                   isinstance(pop.init[variable['name']], int) or \
                   isinstance(pop.init[variable['name']], float) :
                    variable['init'] = pop.init[variable['name']]
                
            for variable in pop.description['variables']:
                if isinstance(pop.init[variable['name']], bool) or \
                   isinstance(pop.init[variable['name']], int) or \
                   isinstance(pop.init[variable['name']], float) :
                    variable['init'] = pop.init[variable['name']]
               
            # Extract RandomDistribution objects
            pop.description['random_distributions'] = extract_randomdist(pop)

            # Extract the spike condition if any
            if 'raw_spike' in pop.description.keys() and 'raw_reset' in pop.description.keys():
                pop.description['spike'] = extract_spike_variable(pop.description)

            # Extract the stop condition if any
            if pop.description.has_key('stop_condition'):
                extract_stop_condition(pop.description)

            # Translate the equations to C++
            for variable in pop.description['variables']:
                eq = variable['transformed_eq']
                untouched={}
                
                # Replace sum(target) with sum(i, rk_target)
                for target in pop.description['targets']:
                    if pop.description['type'] == 'rate': 
                        if target in pop.targets:
                            eq = eq.replace('sum('+target+')', '_sum_'+target )                        
                            untouched['_sum_'+target] = 'sum(i, ' + str(pop.targets.index(target))+')'
                        else: # used in the eq, but not connected
                            eq = eq.replace('sum('+target+')', '0.0' ) 
                    else: # spiking neuron 
                        untouched['g_'+target] = 'g_'+target+'_[i]'
                
                # Extract global operations
                eq, untouched_globs, global_ops = extract_globalops_neuron(variable['name'], eq, pop)
                # Add the untouched variables to the global list
                for name, val in untouched_globs.iteritems():
                    if not untouched.has_key(name):
                        untouched[name] = val
                pop.description['global_operations'] += global_ops
                
                # Extract if-then-else statements
                eq, condition = extract_ite(variable['name'], eq, pop)
                
                # Find the numerical method if any
                if 'implicit' in variable['flags']:
                    method = 'implicit'
                elif 'exponential' in variable['flags']:
                    method = 'exponential'
                else:
                    method = 'explicit'

                # Process the bounds
                if 'min' in variable['bounds'].keys():
                    if isinstance(variable['bounds']['min'], str):
                        translator = Equation(variable['name'], variable['bounds']['min'], 
                                              pop.description['attributes'], 
                                              pop.description['local'], 
                                              pop.description['global'], 
                                              type = 'return',
                                              untouched = untouched.keys())
                        variable['bounds']['min'] = translator.parse().replace(';', '')
                if 'max' in variable['bounds'].keys():
                    if isinstance(variable['bounds']['max'], str):
                        translator = Equation(variable['name'], variable['bounds']['max'], 
                                              pop.description['attributes'], 
                                              pop.description['local'], 
                                              pop.description['global'], 
                                              type = 'return',
                                              untouched = untouched.keys())
                        variable['bounds']['max'] = translator.parse().replace(';', '')
                
                # Analyse the equation
                if condition == []:
                    translator = Equation(variable['name'], eq, 
                                          pop.description['attributes'], 
                                          pop.description['local'], 
                                          pop.description['global'], 
                                          method = method,
                                          untouched = untouched.keys())
                    code = translator.parse()
                else: # An if-then-else statement
                    code = translate_ITE(variable['name'], eq, condition, pop, untouched)

                
                if isinstance(code, str):
                    cpp_eq = code
                    switch = None
                else: # ODE
                    cpp_eq = code[0]
                    switch = code[1]

                # Replace untouched variables with their original name
                for prev, new in untouched.iteritems():
                    if prev.startswith('g_'):
                        cpp_eq = re.sub(r'([^_]+)'+prev, r'\1'+new, cpp_eq)
                        if switch:
                            switch = re.sub(r'^'+prev, new, switch)

                    else:
                        cpp_eq = re.sub(prev, new, cpp_eq)
                        if switch:
                            switch = re.sub(prev, new, switch)

                # Store the result
                variable['cpp'] = cpp_eq # the C++ equation
                variable['switch'] = switch # switch value id ODE
            
        # Generate C++ code for all projection variables 
        for proj in self.projections:
            
            # Actualize initial values
            for variable in proj.description['parameters']:
                if isinstance(proj.init[variable['name']], bool) or \
                   isinstance(proj.init[variable['name']], int) or \
                   isinstance(proj.init[variable['name']], float) :
                    variable['init'] = proj.init[variable['name']]
            for variable in proj.description['variables']:
                if isinstance(proj.init[variable['name']], bool) or \
                   isinstance(proj.init[variable['name']], int) or \
                   isinstance(proj.init[variable['name']], float) :
                    variable['init'] = proj.init[variable['name']]        
             
            # Extract RandomDistribution objects
            proj.description['random_distributions'] = extract_randomdist(proj)
            
            if proj.description['type'] == 'spike': 
                if proj.description['raw_pre_spike']:          
                    proj.description['pre_spike'] = extract_pre_spike_variable(proj)
                
                if proj.description['raw_post_spike']:
                    proj.description['post_spike'] = extract_post_spike_variable(proj)
                        
            # Variables names for the parser which should be left untouched
            untouched = {}   
                       
            # Iterate over all variables
            for variable in proj.description['variables']:
                eq = variable['transformed_eq']
                
                # Replace %(target) by its actual value
                eq = eq.replace('%(target)', proj.target)
                
                # Extract global operations
                eq, untouched_globs, global_ops = extract_globalops_synapse(variable['name'], eq, proj)
                proj.pre.description['global_operations'] += global_ops['pre']
                proj.post.description['global_operations'] += global_ops['post']
                
                # Extract pre- and post_synaptic variables
                eq, untouched_var = extract_prepost(variable['name'], eq, proj)
                
                # Extract if-then-else statements
                eq, condition = extract_ite(variable['name'], eq, proj)
                
                # Add the untouched variables to the global list
                for name, val in untouched_globs.iteritems():
                    if not untouched.has_key(name):
                        untouched[name] = val
                for name, val in untouched_var.iteritems():
                    if not untouched.has_key(name):
                        untouched[name] = val
                        
                # Save the tranformed equation 
                variable['transformed_eq'] = eq
                        
                # Find the numerical method if any
                if 'implicit' in variable['flags']:
                    method = 'implicit'
                elif 'exponential' in variable['flags']:
                    method = 'exponential'
                else:
                    method = 'explicit'

                # Process the bounds
                if 'min' in variable['bounds'].keys():
                    if isinstance(variable['bounds']['min'], str):
                        translator = Equation(variable['name'], variable['bounds']['min'], 
                                              proj.description['attributes'], 
                                              proj.description['local'], 
                                              proj.description['global'], 
                                              type = 'return',
                                              untouched = untouched.keys())
                        variable['bounds']['min'] = translator.parse().replace(';', '')
                if 'max' in variable['bounds'].keys():
                    if isinstance(variable['bounds']['max'], str):
                        translator = Equation(variable['name'], variable['bounds']['max'], 
                                              proj.description['attributes'], 
                                              proj.description['local'], 
                                              proj.description['global'], 
                                              type = 'return',
                                              untouched = untouched.keys())
                        variable['bounds']['max'] = translator.parse().replace(';', '')
                    
                # Analyse the equation
                if condition == []: # Call Equation
                    translator = Equation(variable['name'], eq, proj.description['attributes'], 
                                          proj.description['local'], proj.description['global'], 
                                          method = method, untouched = untouched.keys())
                    code = translator.parse()
                        
                else: # An if-then-else statement
                    code = translate_ITE(variable['name'], eq, condition, proj, untouched)

                if isinstance(code, str):
                    cpp_eq = code
                    switch = None
                else: # ODE
                    cpp_eq = code[0]
                    switch = code[1]

                # Replace untouched variables with their original name
                for prev, new in untouched.iteritems():
                    cpp_eq = cpp_eq.replace(prev, new)     
                
                # Store the result
                variable['cpp'] = cpp_eq # the C++ equation
                variable['switch'] = switch # switch value id ODE
                
            # Translate the psp code if any
            if 'raw_psp' in proj.description.keys():                
                psp = {'eq' : proj.description['raw_psp'].strip() }
                # Replace pre- and post_synaptic variables
                eq = psp['eq']
                eq, untouched = extract_prepost(variable['name'], eq, proj)
                # Extract if-then-else statements
                eq, condition = extract_ite(variable['name'], eq, proj, split=False)
                # Analyse the equation
                if condition == []:
                    translator = Equation('psp', eq, 
                                          proj.description['attributes'], 
                                          proj.description['local'], 
                                          proj.description['global'], 
                                          method = 'explicit', 
                                          untouched = untouched.keys(),
                                          type='return')
                    code = translator.parse()
                else:
                    code = translate_ITE('psp', eq, condition, proj, untouched, split=False)

                # Replace _pre_r_ with (*pre_rates_)[rank_[i]]
                code = code.replace('_pre_r_', '(*pre_rates_)[rank_[i]]')
                # Store the result
                psp['cpp'] = code
                proj.description['psp'] = psp               
        
            # handling delays
            proj.description['csr'] = proj._synapses
        
        # Store the result of analysis for generating the code
        for pop in self.populations:
            # Make sure global operations are generated only once
            glops = []
            for op in pop.description['global_operations']:
                if not op in glops:
                    glops.append(op)
            pop.description['global_operations'] = glops
            # Store the result for generation
            self.analysed_populations[pop.class_name] = pop.description  
            
        for proj in self.projections:
            self.analysed_projections[proj.name] = proj.description  
        return True # success




 
