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
from ANNarchy4.core.Neuron import RateNeuron, SpikeNeuron
from ANNarchy4.core.Synapse import RateSynapse, SpikeSynapse
from ANNarchy4.core.Global import _error, _warning
from ANNarchy4.core.Random import available_distributions
from ANNarchy4.newparser.Equation import Equation

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
        self.targets = []
        
    def analyse(self):
        """ Extracts all the relevant information in the network to prepare code generation."""
        # Iterate over all populations and get basic information
        pop_idx = 1
        for pop in self.populations:
            # Extract information
            description = self._analyse_population(pop)
            # Add the targets to the global list
            self.targets += description['targets']
            # Assign a unique name to the population
            gen_name = 'Population' + str(pop_idx)
            pop_idx += 1
            # Store the result
            self.analysed_populations[gen_name] = description  
            
        # Gather all declared targets            
        self.targets = list(set(self.targets))
            
        # Generate C++ code for all population variables 
        for name, pop in self.analysed_populations.iteritems():
            # Extract RandomDistribution objects
            rk_rand = 0
            random_objects = []
            for variable in pop['variables']:
                eq = variable['eq']
                # Search for all distributions
                for dist in available_distributions:
                    matches = re.findall('(?P<pre>[^\_a-zA-Z0-9.])'+dist+'\(([^()]+)\)', eq)
                    for l, v in matches:
                        # Store its definition
                        desc = {'name': '__rand_' + str(rk_rand) + '_',
                                'definition': dist + '(' + v + ')',
                                'args' : v}
                        rk_rand += 1
                        random_objects.append(desc)
                        # Replace its definition by its temporary name
                        # Problem: when one uses twice the same RD in a single equation (perverse...)
                        eq = eq.replace(desc['definition'], desc['name'])
                        # Add the new variable to the vocabulary
                        pop['attributes'].append(desc['name'])
                        if variable['name'] in pop['local']:
                            pop['local'].append(desc['name'])
                        else: # Why not on a population-wide variable?
                            pop['global'].append(desc['name'])
                variable['eq'] = eq
            pop['random_distributions'] = random_objects
                    
            # Translate the equation to C++
            for variable in pop['variables']:
                # Find the numerical method if any
                if 'implicit' in variable['flags']:
                    method = 'implicit'
                elif 'exponential' in variable['flags']:
                    method = 'exponential'
                else:
                    method = 'explicit'
                # Analyse the equation
                translator = Equation(variable['name'], variable['eq'], pop['attributes'], 
                                      pop['local'], pop['global'], method = method)
                code = translator.parse()
                # Replace sum(target) with sum(i, rk_target)
                for target in self.targets:
                    code = code.replace('sum('+target+')', 'sum(i, '+str(self.targets.index(target))+')')
                # Store the result
                variable['cpp'] = code
                
        # Print the result of the analysis
        pprint(self.analysed_populations)

                
        # Iterate over all projections and get basic information
        proj_idx = 1
        for proj in self.projections:
            # Extract information
            description = self._analyse_projection(proj)            
            # Assign a unique name to the population
            gen_name = 'Projection' + str(proj_idx)
            proj_idx += 1
            # Store the result
            self.analysed_projections[gen_name] = description
            
            
        # Generate C++ code for all projection variables 
        for name, proj in self.analysed_projections.iteritems():
            # Variables names for the parser which should be left untouched
            untouched = {}
            
            # Extract RandomDistribution objects
            rk_rand = 0
            random_objects = []
            for variable in proj['variables']:
                eq = variable['eq']
                # Search for all distributions
                for dist in available_distributions:
                    matches = re.findall('(?P<pre>[^\_a-zA-Z0-9.])'+dist+'\(([^()]+)\)', eq)
                    for l, v in matches:
                        # Store its definition
                        desc = {'name': '__rand_' + str(rk_rand) + '_',
                                'definition': dist + '(' + v + ')',
                                'args' : v}
                        rk_rand += 1
                        random_objects.append(desc)
                        # Replace its definition by its temporary name
                        # Problem: when one uses twice the same RD in a single equation (perverse...)
                        eq = eq.replace(desc['definition'], desc['name'])
                        # Add the new variable to the vocabulary
                        proj['attributes'].append(desc['name'])
                        if variable['name'] in proj['local']:
                            proj['local'].append(desc['name'])
                        else: # Why not on a population-wide variable?
                            proj['global'].append(desc['name'])
                variable['eq'] = eq
            proj['random_distributions'] = random_objects
            
            # Replace %(target) by its actual value
            for variable in proj['variables']:
                variable['eq'] = variable['eq'].replace('%(target)', proj['target'])
            
            # Replace pre- and post_synaptic variables
            for variable in proj['variables']:
                eq = variable['eq']
                pre_matches = re.findall('pre.([a-zA-Z0-9]+)', eq)
                post_matches = re.findall('post.([a-zA-Z0-9]+)', eq)
                # Check if a global variable depends on pre
                if len(pre_matches) > 0 and variable['name'] in proj['global']:
                    _error(eq + '\nA postsynaptic variable can not depend on pre.'+pre_matches[0])
                    exit(0)
                # Replace all pre.* occurences with a temporary variable
                for var in list(set(pre_matches)):
                    prepop = self._find_population_description(proj['pre'])
                    if var == 'sum': # pre.sum(exc)
                        pass
                    elif var in prepop['attributes']:
                        target = 'pre.' + var
                        eq = eq.replace(target, '_pre_'+var+'_')
                        untouched['_pre_'+var+'_'] = ' pre_population_->getSingle'+var.capitalize()+'( rank_[i] ) '
                    else:
                        _error(eq+'\nPopulation '+proj['pre']+' has no attribute '+var+'.')
                        exit(0)
                # Replace all post.* occurences with a temporary variable
                for var in list(set(post_matches)):
                    prepop = self._find_population_description(proj['post'])
                    if var == 'sum': # pre.sum(exc)
                        pass
                    elif var in prepop['attributes']:
                        target = 'post.' + var
                        eq = eq.replace(target, '_post_'+var+'_')
                        untouched['_post_'+var+'_'] = ' post_population_->getSingle'+var.capitalize()+'(  post_neuron_rank_ ) '
                    else:
                        _error(eq+'\nPopulation '+proj['post']+' has no attribute '+var+'.')
                        exit(0)
                variable['eq'] = eq
                    
            # Translate the equation to C++
            for variable in proj['variables']:
                # Find the numerical method if any
                if 'implicit' in variable['flags']:
                    method = 'implicit'
                elif 'exponential' in variable['flags']:
                    method = 'exponential'
                else:
                    method = 'explicit'
                # Analyse the equation
                translator = Equation(variable['name'], variable['eq'], proj['attributes'], 
                                      proj['local'], proj ['global'], method = method, untouched = untouched.keys())
                code = translator.parse()
                # Replace untouched variables with their original name
                for prev, next in untouched.iteritems():
                    code = code.replace(prev, next)
                # Store the result
                variable['cpp'] = code
            
        
        # Print the result of the analysis
        pprint(self.analysed_projections)
            
        return True # success
        
    def _analyse_population(self, pop):
        """ Performs the analysis for a single population."""
        # Identify the population type
        pop_type = 'rate' if isinstance(pop.neuron_type, RateNeuron) else 'spike'
        # Store basic information
        description = {
            'name': pop.name,
            'type': pop_type,
            'raw_parameters': pop.neuron_type.parameters,
            'raw_equations': pop.neuron_type.equations,
            'raw_functions': pop.neuron_type.functions
        }
        if pop_type == 'spike': # Additionally store reset and spike
            description['raw_reset'] = pop.neuron_type.reset
            description['raw_spike'] = pop.neuron_type.spike
        # Extract parameters and variables names
        parameters = self._extract_parameters(pop.neuron_type.parameters)
        variables = self._extract_variables(pop.neuron_type.equations)
        # Build lists of all attributes (param+var), which are local or global
        attributes, local_var, global_var = self._get_attributes(parameters, variables)
        # Extract all targets
        targets = self._extract_targets(variables)
        # Add this info to the description
        description['parameters'] = parameters
        description['variables'] = variables
        description['attributes'] = attributes
        description['local'] = local_var
        description['global'] = global_var
        description['targets'] = targets
        return description

    def _analyse_projection(self, proj):  
        """ Performs the analysis for a single projection."""      
        # Identify the synapse type
        proj_type = 'rate' if isinstance(proj.synapse, RateSynapse) else 'spike'
        # Store basic information
        description = {
            'pre': proj.pre.name,
            'post': proj.post.name,
            'target': proj.target,
            'type': proj_type,
            'raw_parameters': proj.synapse.parameters,
            'raw_equations': proj.synapse.equations,
            'raw_functions': proj.synapse.functions
        }
        if proj_type == 'spike': # Additionally store pre_spike and post_spike
            description['raw_pre_spike'] = proj.synapse.pre_spike
            description['raw_post_spike'] = proj.synapse.post_spike
        # Extract parameters and variables names
        parameters = self._extract_parameters(proj.synapse.parameters)
        variables = self._extract_variables(proj.synapse.equations)
        # Build lists of all attributes (param+var), which are local or global
        attributes, local_var, global_var = self._get_attributes(parameters, variables)
        # Add this info to the description
        description['parameters'] = parameters
        description['variables'] = variables
        description['attributes'] = attributes
        description['local'] = local_var
        description['global'] = global_var
        return description
        
    def _extract_parameters(self, description):
        """ Extracts all variable information from a multiline description."""
        parameters = []
        # Split the multilines into individual lines
        parameter_list = self._prepare_string(description)
        # Analyse all variables
        for definition in parameter_list:
            # Check if there are flags after the : symbol
            equation, constraint = self._split_equation(definition)
            # Extract the name of the variable
            name = self._extract_name(equation)
            if name == '_undefined':
                exit(0)
            # Process the flags if any
            bounds, flags = self._extract_flags(constraint)
            # Get the type of the variable (float/int/bool)
            if 'int' in flags:
                ctype = 'int'
            elif 'bool' in flags:
                ctype = 'bool'
            else:
                ctype = 'DATA_TYPE'
            # For parameters, the initial value can be given in the equation
            if 'init' in bounds.keys(): # if init is provided, it wins
                init = bounds['init']
                if ctype == 'bool':
                    if init in ['false', 'False', '0']:
                        init = 'false'
                    elif init in ['true', 'True', '1']:
                        init = 'true'
                else:
                    init = ctype + '(' + init + ')'
            elif '=' in equation: # the value is in the equation
                init = equation.split('=')[1].strip()
                if init in ['false', 'False']:
                    init = 'false'
                    ctype = 'bool'
                elif init in ['true', 'True']:
                    init = 'true'
                    ctype = 'bool'
                else:
                    init = ctype + '(' + init + ')'
            else: # Nothing is given: baseline : population
                if ctype == 'bool':
                    init = 'false'
                elif ctype == 'int':
                    init = '0'
                elif ctype == 'DATA_TYPE':
                    init = '0.0'
                
            # Store the result
            desc = {'name': name,
                    'eq': equation,
                    'bounds': bounds,
                    'flags' : flags,
                    'ctype' : ctype,
                    'init' : init }
            parameters.append(desc)              
        return parameters
        
    def _extract_variables(self, description):
        """ Extracts all variable information from a multiline description."""
        variables = []
        # Split the multilines into individual lines
        variable_list = self._prepare_string(description)
        # Analyse all variables
        for definition in variable_list:
            # Check if there are flags after the : symbol
            equation, constraint = self._split_equation(definition)
            # Extract the name of the variable
            name = self._extract_name(equation)
            if name == '_undefined':
                exit(0)
            # Process the flags if any
            bounds, flags = self._extract_flags(constraint)
            # Get the type of the variable (float/int/bool)
            if 'int' in flags:
                ctype = 'int'
            elif 'bool' in flags:
                ctype = 'bool'
            else:
                ctype = 'DATA_TYPE'
            # Get the init value if declared
            if 'init' in bounds.keys():
                init = bounds['init']
                if ctype == 'bool':
                    if init in ['false', 'False', '0']:
                        init = 'false'
                    elif init in ['true', 'True', '1']:
                        init = 'true'
                else:
                    init = ctype + '(' + init + ')'
            else: # Default = 0 according to ctype
                if ctype == 'bool':
                    init = 'false'
                elif ctype == 'int':
                    init = '0'
                elif ctype == 'DATA_TYPE':
                    init = '0.0'
            # Store the result
            desc = {'name': name,
                    'eq': equation,
                    'bounds': bounds,
                    'flags' : flags,
                    'ctype' : ctype,
                    'init' : init }
            variables.append(desc)              
        return variables
        
    def _prepare_string(self, stream):
        """ Splits up the several equations, remove comments and unneeded spaces or tabs."""
        expr_set = []        
        # replace the ,,, by empty space and split the result up
        tmp_set = re.sub('\s+\.\.\.\s+', ' ', stream).split('\n')
        for expr in tmp_set:
            expr = re.sub('\#[\s\S]+', ' ', expr)   # remove comments
            expr = re.sub('\s+', ' ', expr)     # remove additional tabs etc.
            if expr == ' ' or len(expr)==0: # through beginning line breaks or something similar empty strings are contained in the set
                continue           
            expr_set.append(''.join(expr))        
        return expr_set
        
    def _split_equation(self, definition):
        " Splits a description into equation and flags."
        try:
            equation, constraint = definition.split(':')
            equation = equation.strip()
            constraint = constraint.strip()
        except ValueError:
            equation = definition.strip() # there are no constraints
            constraint = None
        finally:
            return equation, constraint
        
    def _extract_name(self, equation):
        " Extracts the name of a parameter/variable by looking the left term of an equation."
        equation = equation.replace(' ','')
        try:
            name = equation.split('=')[0]
        except: # No equal sign. Eg: baseline : init=0.0
            return equation.strip()
        # Search for increments
        operators = ['+=', '-=', '*=', '/=']
        for op in operators:
            if op in equation: 
                return equation.split(op)[0]        
        # Search for any operation in the left side
        operators = ['+', '-', '*', '/']
        ode = False
        for op in operators:
            if not name.find(op) == -1: 
                ode = True
        if not ode: # variable name is alone on the left side
            return name
        # ODE: the variable name is between d and /dt
        name = re.findall("(?<=d)[\w\s]+(?=/dt)", name)
        if len(name) == 1:
            return name[0].strip()
        else:
            _error('No variable name can be found in ' + equation)
            return '_undefined'
                
    def _extract_flags(self, constraint):
        """ Extracts from all attributes given after : which are bounds (eg min=0.0 or init=0.1) 
            and which are flags (eg postsynaptic, implicit...).
        """
        bounds = {}
        flags = []
        # Check if there are constraints at all
        if not constraint:
            return bounds, flags
        # Split according to ','
        for con in constraint.split(','):
            try: # bound of the form key = val
                key, value = con.split('=')
                bounds[key.strip()] = value.strip()
            except ValueError: # No equal sign = flag
                flags.append(con.strip())
        return bounds, flags
        
    def _get_attributes(self, parameters, variables):
        """ Returns a list of all attributes names, plus the lists of local/global variables."""
        attributes = []; local_var = []; global_var = []
        for p in parameters + variables:
            attributes.append(p['name'])
            if 'population' in p['flags'] or 'postsynaptic' in p['flags']:
                global_var.append(p['name'])
            else:
                local_var.append(p['name'])
        return attributes, local_var, global_var
    
    def _extract_targets(self, variables):
        targets = []
        for var in variables:
            code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])sum\(([^()]+)\)', var['eq'])
            for l, t in code:
                targets.append(t)
        return list(set(targets))
    
        
    def _find_population_description(self, name):
        " Returns the populations description corresponding to the population name"
        for n, desc in self.analysed_populations.iteritems():
            if desc['name'] == name:
                return self.analysed_populations[n]
        return None
            
        
    def generate(self):
        """ Generates C++ code once the analysis is performed."""
        exit(0)
