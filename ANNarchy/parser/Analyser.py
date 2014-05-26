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
from ANNarchy.core.Global import _error, _warning, authorized_keywords
from ANNarchy.core.Random import available_distributions
from ANNarchy.parser.Equation import Equation
from ANNarchy.parser.Function import FunctionParser

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
            pop.description['random_distributions'] = _extract_randomdist(pop)
                   
            if 'raw_spike' in pop.description.keys() and 'raw_reset' in pop.description.keys():
                pop.description['spike'] = _extract_spike_variable(pop.description)

            # Translate the equation to C++
            for variable in pop.description['variables']:
                eq = variable['transformed_eq']
                untouched={}
                
                # Replace sum(target) with sum(i, rk_target)
                for target in pop.description['targets']:
                    if target in pop.targets:
                        eq = eq.replace('sum('+target+')', '_sum_'+target )                        
                        untouched['_sum_'+target] = 'sum(i, ' + str(pop.targets.index(target))+')'
                    else: # used in the eq, but not connected
                        eq = eq.replace('sum('+target+')', '0.0' ) 
                
                # Extract global operations
                eq, untouched_globs, global_ops = _extract_globalops_neuron(variable['name'], eq, pop)
                # Add the untouched variables to the global list
                for name, val in untouched_globs.iteritems():
                    if not untouched.has_key(name):
                        untouched[name] = val
                pop.description['global_operations'] += global_ops
                
                # Extract if-then-else statements
                eq, condition = _extract_ite(variable['name'], eq, pop)
                
                # Find the numerical method if any
                if 'implicit' in variable['flags']:
                    method = 'implicit'
                elif 'exponential' in variable['flags']:
                    method = 'exponential'
                else:
                    method = 'explicit'
                
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
                    code = self._translate_ITE(variable['name'], eq, condition, pop, untouched)
                    
                
                # Replace untouched variables with their original name
                for prev, new in untouched.iteritems():
                    code = code.replace(prev, new) 
                
                # Store the result
                variable['cpp'] = code
            
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
            proj.description['random_distributions'] = _extract_randomdist(proj)
            
            if proj.description['type'] == 'spike': 
                if proj.description['raw_pre_spike']:          
                    proj.description['pre_spike'] = _extract_pre_spike_variable(proj.description)
                
                if proj.description['raw_post_spike']:
                    proj.description['post_spike'] = _extract_post_spike_variable(proj.description)
                        
            # Variables names for the parser which should be left untouched
            untouched = {}   
                       
            # Iterate over all variables
            for variable in proj.description['variables']:
                eq = variable['transformed_eq']
                
                # Replace %(target) by its actual value
                eq = eq.replace('%(target)', proj.target)
                
                # Extract global operations
                eq, untouched_globs, global_ops = _extract_globalops_synapse(variable['name'], eq, proj)
                proj.pre.description['global_operations'] += global_ops['pre']
                proj.post.description['global_operations'] += global_ops['post']
                
                # Extract pre- and post_synaptic variables
                eq, untouched_var = _extract_prepost(variable['name'], eq, proj)
                
                # Extract if-then-else statements
                eq, condition = _extract_ite(variable['name'], eq, proj)
                
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
                    
                # Analyse the equation
                if condition == []: # Call Equation
                    translator = Equation(variable['name'], eq, proj.description['attributes'], 
                                          proj.description['local'], proj.description['global'], 
                                          method = method, untouched = untouched.keys())
                    code = translator.parse()
                        
                else: # An if-then-else statement
                    code = self._translate_ITE(variable['name'], eq, condition, proj, untouched)
                       
                # Replace untouched variables with their original name
                for prev, new in untouched.iteritems():
                    code = code.replace(prev, new)     
                
                # Store the result
                variable['cpp'] = code
                
            # Translate the psp code if any
            if 'raw_psp' in proj.description.keys():                
                psp = {'eq' : proj.description['raw_psp'].strip() }
                # Replace pre- and post_synaptic variables
                eq = psp['eq']
                eq, untouched = _extract_prepost(variable['name'], eq, proj)
                # Extract if-then-else statements
                eq, condition = _extract_ite(variable['name'], eq, proj, split=False)
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
                    code = self._translate_ITE('psp', eq, condition, proj, untouched, split=False)

                # Replace _pre_rate_ with (*pre_rates_)[rank_[i]]
                code = code.replace('_pre_rate_', '(*pre_rates_)[rank_[i]]')
                # Store the result
                psp['cpp'] = code
                proj.description['psp'] = psp               
        
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
    
    def _translate_ITE(self, name, eq, condition, proj, untouched, split=True):
        " Recursively processes the different parts of an ITE statement"
        def process_ITE(condition):
            if_statement = condition[0]
            then_statement = condition[1]
            else_statement = condition[2]
            if_code = Equation(name, if_statement, proj.description['attributes'], 
                              proj.description['local'], proj.description['global'], 
                              method = 'explicit', untouched = untouched.keys(),
                              type='cond').parse()
            if isinstance(then_statement, list): # nested conditional
                then_code =  process_ITE(then_statement)
            else:
                then_code = Equation(name, then_statement, proj.description['attributes'], 
                              proj.description['local'], proj.description['global'], 
                              method = 'explicit', untouched = untouched.keys(),
                              type='return').parse().split(';')[0]
            if isinstance(else_statement, list): # nested conditional
                else_code =  process_ITE(else_statement)
            else:
                else_code = Equation(name, else_statement, proj.description['attributes'], 
                              proj.description['local'], proj.description['global'], 
                              method = 'explicit', untouched = untouched.keys(),
                              type='return').parse().split(';')[0]
                              
            code = '(' + if_code + ' ? ' + then_code + ' : ' + else_code + ')'
            return code
              
        if split:
            # Main equation, where the right part is __conditional__
            translator = Equation(name, eq, proj.description['attributes'], 
                                  proj.description['local'], proj.description['global'], 
                                  method = 'explicit', untouched = untouched.keys())
            code = translator.parse() 
        else:
            code = '__conditional__'
        # Process the ITE
        itecode =  process_ITE(condition)
        # Replace
        code = code.replace('__conditional__', itecode)
        return code

def _extract_ite(name, eq, proj, split=True):
    """ Extracts if-then-else statements and processes them.
    
    If-then-else statements must be of the form:
    
    .. code-block:: python
    
        variable = if condition: ...
                       val1 ...
                   else: ...
                       val2
                       
    Conditional statements can be nested, but they should return only one value!
    """
    
    def transform(code):
        " Transforms the code into a list of lines."
        res = []
        items = []
        for arg in code.split(':'):
            items.append( arg.strip())
        for i in range(len(items)):
            if items[i].startswith('if '):
                res.append( items[i].strip() )
            elif items[i].endswith('else'):
                res.append(items[i].split('else')[0].strip() )
                res.append('else' )
            else: # the last then
                res.append( items[i].strip() )    
        return res
        
        
    def parse(lines):
        " Recursive analysis of if-else statements"
        result = []
        while lines:
            if lines[0].startswith('if'):
                block = [lines.pop(0).split('if')[1], parse(lines)]
                if lines[0].startswith('else'):
                    lines.pop(0)
                    block.append(parse(lines))
                result.append(block)
            elif not lines[0].startswith(('else')):
                result.append(lines.pop(0))
            else:
                break
        return result[0]
    # If no if, no need to go further
    if not 'if ' in eq:
        return eq, []
    # Process the equation            
    condition = []   
    # Eventually split around =
    if split:
        left, right =  eq.split('=', 1)
    else:
        left = ''
        right = eq
    nb_then = len(re.findall(':', right))
    nb_else = len(re.findall('else', right))
    # The equation contains a conditional statement
    if nb_then > 0:
        # A if must be right after the equal sign
        if not right.strip().startswith('if'):
            _error(eq, '\nThe right term must directly start with a if statement.')
            exit(0)
        # It must have the same number of : and of else
        if not nb_then == 2*nb_else:
            _error(eq, '\nConditional statements must use both : and else.')
            exit(0)
        multilined = transform(right)
        condition = parse(multilined)
        right = ' __conditional__ '
        if split:
            eq = left + '=' + right
    return eq, condition

def _extract_randomdist(pop):
    " Extracts RandomDistribution objects from all variables"
    rk_rand = 0
    random_objects = []
    for variable in pop.description['variables']:
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
                pop.description['attributes'].append(desc['name'])
                if variable['name'] in pop.description['local']:
                    pop.description['local'].append(desc['name'])
                else: # Why not on a population-wide variable?
                    pop.description['global'].append(desc['name'])
        variable['transformed_eq'] = eq
        
    return random_objects
    
def _extract_globalops_neuron(name, eq, pop):
    """ Replaces global operations (mean(r), etc)  with arbitrary names and 
    returns a dictionary of changes.
    """
    untouched = {}    
    globs = []  
    glop_names = ['min', 'max', 'mean']
    
    for op in glop_names:
        matches = re.findall('([^\w]*)'+op+'\(([\w]*)\)', eq)
        for pre, var in matches:
            if var in pop.description['local']:
                globs.append({'function': op, 'variable': var})
                oldname = op + '(' + var + ')'
                newname = '__' + op + '_' + var 
                eq = eq.replace(oldname, newname)
                untouched[newname] = ' this->get'+op.capitalize()+var.capitalize()+'() '
            else:
                _error(eq+'\nPopulation '+pop.name+' has no local attribute '+var+'.')
                exit(0)
    return eq, untouched, globs
    
def _extract_globalops_synapse(name, eq, proj):
    """ Replaces global operations (mean(pre.r), etc)  with arbitrary names and 
    returns a dictionary of changes.
    """
    untouched = {}    
    globs = {'pre' : [],
             'post' : [] }   
    glop_names = ['min', 'max', 'mean']
    
    for op in glop_names:
        pre_matches = re.findall('([^a-zA-Z0-9.])'+op+'\(\s*pre\.([a-zA-Z0-9]+)\s*\)', eq)
        post_matches = re.findall('([^a-zA-Z0-9.])'+op+'\(\s*post\.([a-zA-Z0-9]+)\s*\)', eq)
        # Check if a global variable depends on pre
        if len(pre_matches) > 0 and name in proj.description['global']:
            _error(eq + '\nA postsynaptic variable can not depend on pre.' + pre_matches[0])
            exit(0)
        for pre, var in pre_matches:
            if var in proj.pre.attributes:
                globs['pre'].append({'function': op, 'variable': var})
                oldname = op + '(pre.' + var + ')'
                newname = '__pre_' + op + '_' + var 
                eq = eq.replace(oldname, newname)
                untouched[newname] = ' pre_population_->get'+op.capitalize()+var.capitalize()+'() '
            else:
                _error(eq+'\nPopulation '+proj.name+' has no attribute '+var+'.')
                exit(0)
        for pre, var in post_matches:
            if var in proj.pre.attributes:
                globs['post'].append({'function': op, 'variable': var})
                oldname = op + '(post.' + var + ')'
                newname = '__post_' + op + '_' + var 
                eq = eq.replace(oldname, newname)
                untouched[newname] = ' post_population_->get'+op.capitalize()+var.capitalize()+'() '
            else:
                _error(eq+'\nPopulation '+proj.pre.name+' has no attribute '+var+'.')
                exit(0)
    return eq, untouched, globs
    
def _extract_prepost(name, eq, proj):
    " Replaces pre.var and post.var with arbitrary names and returns a dictionary of changes."
    untouched = {}                
    pre_matches = re.findall(r'pre\.([a-zA-Z0-9_]+)', eq)
    post_matches = re.findall(r'post\.([a-zA-Z0-9_]+)', eq)
    # Check if a global variable depends on pre
    if len(pre_matches) > 0 and name in proj.description['global']:
        _error(eq + '\nA postsynaptic variable can not depend on pre.' + pre_matches[0])
        exit(0)
    # Replace all pre.* occurences with a temporary variable
    for var in list(set(pre_matches)):
        if var == 'sum': # pre.sum(exc)
            def idx_target(val):
                try:
                    idx = proj.pre.targets.index(val.group(1))
                except:
                    _warning('The target ' + val.group(1) + ' does not exist in the population ' + proj.pre.name)
                    return '0.0'
                rep = '_pre_sum_' + str(idx)
                untouched[rep] = ' pre_population_->sum(i, '+str(idx)+' ) '
                return rep
            eq = re.sub(r'pre\.sum\(([a-zA-Z]+)\)', idx_target, eq)
        elif var in proj.pre.attributes:
            target = 'pre.' + var
            eq = eq.replace(target, ' _pre_'+var+'_ ')
            if var=='r':
                untouched['_pre_r_'] = ' (*pre_rates_)[ rank_[i] ] '
            else:
                untouched['_pre_'+var+'_'] = ' pre_population_->get_single_'+var+'( rank_[i] ) '
        else:
            _error(eq+'\nPopulation '+proj.pre.description['name']+' has no attribute '+var+'.')
            exit(0)
    # Replace all post.* occurences with a temporary variable
    for var in list(set(post_matches)):
        if var == 'sum': # post.sum(exc)
            def idx_target(val):
                try:
                    idx = proj.post.targets.index(val.group(1))
                except:
                    _warning('The target ' + val.group(1) + ' does not exist in the population ' + proj.post.name)
                    return '0.0'
                rep = '_post_sum_' + str(idx)
                untouched[rep] = ' post_population_->sum(post_neuron_rank_, '+str(idx)+' ) '
                return rep
            eq = re.sub(r'post\.sum\(([a-zA-Z]+)\)', idx_target, eq)
        elif var in proj.post.attributes:
            target = 'post.' + var
            eq = eq.replace(target, ' _post_'+var+'_ ')
            if var.strip() == 'r':
                #the firing rates are solved in a slightly different way
                untouched['_post_r_'] = ' post_r_ '
            else:
                untouched['_post_'+var+'_'] = ' post_population_->get_single_'+var+'(  post_neuron_rank_ ) '
        else:
            _error(eq+'\nPopulation '+proj.post.description['name']+' has no attribute '+var+'.')
            exit(0)
    return eq, untouched
            
def analyse_population(pop):
    """ Performs the initial analysis for a single population."""
    # Identify the population type
    pop_type = 'rate' if isinstance(pop.neuron_type, RateNeuron) else 'spike'
    # Store basic information
    description = {
        'pop': pop,
        'name': pop.name,
        'class': pop.class_name,
        'type': pop_type,
        'raw_parameters': pop.neuron_type.parameters,
        'raw_equations': pop.neuron_type.equations,
        'raw_functions': pop.neuron_type.functions,
    }
    if pop_type == 'spike': # Additionally store reset and spike
        description['raw_reset'] = pop.neuron_type.reset
        description['raw_spike'] = pop.neuron_type.spike
        description['refractory'] = pop.neuron_type.refractory
        
    # Extract parameters and variables names
    parameters = _extract_parameters(pop.neuron_type.parameters, pop.neuron_type.extra_values)
    variables = _extract_variables(pop.neuron_type.equations)
    # Extract functions
    functions = _extract_functions(pop.neuron_type.functions, False)
    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = _get_attributes(parameters, variables)
    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error(pop.name, ': attributes must be declared only once.', attributes)
        exit(0)
    # Extract all targets
    targets = _extract_targets(variables)
    # Add this info to the description
    description['parameters'] = parameters
    description['variables'] = variables
    description['functions'] = functions
    description['attributes'] = attributes
    description['local'] = local_var
    description['global'] = global_var
    description['targets'] = targets
    description['global_operations'] = []
    return description

def analyse_projection(proj):  
    """ Performs the analysis for a single projection.""" 
    #proj_type = 'rate' if isinstance(proj.synapse_type, RateSynapse) else 'spike'

    # Identify the synapse type depending on pre and postsynaptic population
    if proj.post.description['type'] == 'rate':
        proj_type = 'rate'
        if not isinstance(proj.synapse_type, RateSynapse):
            print proj.synapse_type
            _error("SpikeSynapse attached to rate coded populations.")        
    else:
        proj_type = 'spike'
        if isinstance(proj.synapse_type, RateSynapse):
            _error("RateSynapse attached to spike coded populations.")        

    # Store basic information
    description = {
        'pre': proj.pre.name,
        'pre_class': proj.pre.class_name,
        'post': proj.post.name,
        'post_class': proj.post.class_name,
        'target': proj.target,
        'type': proj_type,
        'raw_parameters': proj.synapse_type.parameters,
        'raw_equations': proj.synapse_type.equations,
        'raw_functions': proj.synapse_type.functions
    }

    if proj_type == 'spike': # Additionally store pre_spike and post_spike
        proj.pre.sources.append(proj.target) 
        if proj.synapse_type.pre_spike:
            description['raw_pre_spike'] = proj.synapse_type.pre_spike
        else: # pre_spike not defined, but other fields yes
            description['raw_pre_spike'] = "g_target += w"
        description['raw_post_spike'] = proj.synapse_type.post_spike

    if proj.synapse_type.psp:
        description['raw_psp'] = proj.synapse_type.psp

    # Extract parameters and variables names
    parameters = _extract_parameters(proj.synapse_type.parameters, proj.synapse_type.extra_values)
    variables = _extract_variables(proj.synapse_type.equations)
    # Extract functions
    functions = _extract_functions(proj.synapse_type.functions, False)
    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = _get_attributes(parameters, variables)
    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error(proj.name, ': attributes must be declared only once.', attributes)
        exit(0)
    
    # extend the list of parameters by rank and delay
    parameters += [ { 'name': 'rank', 'init': 0, 'ctype': 'int', 'flags': [] }, 
                    { 'name': 'delay', 'init': 0, 'ctype': 'int', 'flags': [] } 
                  ]
    
    # if no synapse was set during creation of the projection
    # the variable w is not attached to the set of variables
    found = False
    for var in variables+parameters:
        if var['name'] == "w":
            found = True
    
    # w is not declared so far, so we simply add a parameter
    if not found:
        parameters += [ { 'name': 'w', 'init': 0, 'ctype': 'DATA_TYPE', 'flags': [] } ]
        
    # Add this info to the description
    description['parameters'] = parameters
    description['variables'] = variables
    description['functions'] = functions
    description['attributes'] = attributes
    description['local'] = local_var
    description['global'] = global_var
    description['global_operations'] = []
    
    return description            

def _extract_parameters(description, extra_values):
    """ Extracts all variable information from a multiline description."""
    parameters = []
    # Split the multilines into individual lines
    parameter_list = _prepare_string(description)
    
    # Analyse all variables
    for definition in parameter_list:
        # Check if there are flags after the : symbol
        equation, constraint = _split_equation(definition)
        # Extract the name of the variable
        name = _extract_name(equation)
        if name == '_undefined':
            exit(0)
        # Process the flags if any
        bounds, flags = _extract_flags(constraint)

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
                    init = False
                elif init in ['true', 'True', '1']:
                    init = True
            elif ctype == 'int':
                init = int(init)
            else:
                init = float(init)
        elif '=' in equation: # the value is in the equation
            init = equation.split('=')[1].strip()
            
            if init in ['false', 'False']:
                init = False
                ctype = 'bool'
            elif init in ['true', 'True']:
                init = True
                ctype = 'bool'
            else:
                try:
                    init = eval(ctype.replace('DATA_TYPE', 'float') + '(' + init + ')')
                except:
                    try:
                        init = eval('int(' + init + ')')
                    except:
                        var = init.replace("'","")
                        init = extra_values[var]
                    
        else: # Nothing is given: baseline : population
            if ctype == 'bool':
                init = False
            elif ctype == 'int':
                init = 0
            elif ctype == 'DATA_TYPE':
                init = 0.0
            
        # Store the result
        desc = {'name': name,
                'eq': equation,
                'bounds': bounds,
                'flags' : flags,
                'ctype' : ctype,
                'init' : init}
        parameters.append(desc)              
    return parameters
    
def _extract_variables(description):
    """ Extracts all variable information from a multiline description."""
    variables = []
    # Split the multilines into individual lines
    variable_list = _process_equations(description)
    # Analyse all variables
    for definition in variable_list:
        # Retrieve the name, equation and constraints for the variable
        equation = definition['eq']
        constraint = definition['constraint']
        name = definition['name']
        if name == '_undefined':
            exit(0)
        # Process the flags if any
        bounds, flags = _extract_flags(constraint)
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
                    init = False
                elif init in ['true', 'True', '1']:
                    init = True
            elif ctype == 'int':
                init = int(init)
            else:
                init = float(init)
        else: # Default = 0 according to ctype
            if ctype == 'bool':
                init = False
            elif ctype == 'int':
                init = 0
            elif ctype == 'DATA_TYPE':
                init = 0.0
        # Store the result
        desc = {'name': name,
                'eq': equation,
                'bounds': bounds,
                'flags' : flags,
                'ctype' : ctype,
                'init' : init }
        variables.append(desc)              
    return variables        
    
def _extract_functions(description, local_global=False):
    """ Extracts all functions from a multiline description."""
    if not description:
        return [] 
    # Split the multilines into individual lines
    function_list = _process_equations(description)
    # Process each function
    functions = []
    for f in function_list:
        eq = f['eq']
        var_name, content = eq.split('=', 1)
        # Extract the name of the function
        func_name = var_name.split('(', 1)[0].strip()
        # Extract the arguments
        arguments = (var_name.split('(', 1)[1].split(')')[0]).split(',')
        arguments = [arg.strip() for arg in arguments]
        # Extract their types
        types = f['constraint']
        if types == '':
            return_type = 'DATA_TYPE'
            arg_types = ['DATA_TYPE' for a in arguments]
        else:
            types = types.split(',')
            return_type = types[0].strip()
            if return_type == 'float':
                return_type = 'DATA_TYPE'
            arg_types = ['DATA_TYPE' if arg.strip() == 'float' else arg.strip() for arg in types[1:]]
        if not len(arg_types) == len(arguments):
            _error('You must specify exactly the types of return value and arguments in ' + eq)
            exit(0)
        arg_line = ""
        for i in range(len(arguments)):
            arg_line += arg_types[i] + " " + arguments[i]
            if not i == len(arguments) -1:
                arg_line += ', '
        # Process the content
        eq2, condition = _extract_ite('', content, None, split=False)
        if condition == []:
            parser = FunctionParser(content, arguments)
            parsed_content = parser.parse()
        else:
            parser = FunctionParser(content, arguments)
            parsed_content = parser.process_ITE(condition)
        # Create the one-liner
        fdict = {'name': func_name, 'args': arguments, 'content': content, 'return_type': return_type, 'arg_types': arg_types, 'parsed_content': parsed_content, 'arg_line': arg_line}
        if not local_global: # local to a class
            oneliner = """%(return_type)s %(name)s (%(arg_line)s) {return %(parsed_content)s ;};
""" % fdict
        else: # global
            oneliner = """inline %(return_type)s %(name)s (%(arg_line)s) {return %(parsed_content)s ;};
""" % fdict
        fdict['cpp'] = oneliner
        functions.append(fdict)
    return functions
    
    
def _get_attributes(parameters, variables):
    """ Returns a list of all attributes names, plus the lists of local/global variables."""
    attributes = []; local_var = []; global_var = []
    for p in parameters + variables:
        attributes.append(p['name'])
        if 'population' in p['flags'] or 'postsynaptic' in p['flags']:
            global_var.append(p['name'])
        else:
            local_var.append(p['name'])
    return attributes, local_var, global_var

def _extract_targets(variables):
    targets = []
    for var in variables:
        code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])sum\(([^()]+)\)', var['eq'])
        for l, t in code:
            targets.append(t)
    return list(set(targets))

def _extract_spike_variable(pop_desc):
    spike_name = _extract_condition_name(pop_desc['raw_spike'].strip())
    translator = Equation('raw_spike_cond', pop_desc['raw_spike'], 
                          pop_desc['attributes'], 
                          pop_desc['local'], 
                          pop_desc['global'], 
                          type = 'cond')
    raw_spike_code = translator.parse()
    
    raw_reset_code = ''
    if pop_desc.has_key('raw_reset') and pop_desc['raw_reset']:
        for var in _prepare_string(pop_desc['raw_reset']):
            name = _extract_name(var)
            translator = Equation(name, var, 
                                  pop_desc['attributes'], 
                                  pop_desc['local'], 
                                  pop_desc['global'])
            raw_reset_code += translator.parse() +'\n'
    
    return { 'name': spike_name, 'spike_cond': raw_spike_code, 'spike_reset': raw_reset_code}

def _extract_pre_spike_variable(proj_desc):
    pre_spike_var = []
    # For all variables influenced by a presynaptic spike
    for var in _prepare_string(proj_desc['raw_pre_spike']):
        # Get its name
        name = _extract_name(var)
        # Replace target by its value
        post_target = False
        if name == "g_target":
            name = name.replace("target", proj_desc['target'])
            post_target = True
            
        translator = Equation(name, var, 
                              proj_desc['attributes'] + [name], 
                              proj_desc['local'] + [name], 
                              proj_desc['global'],
                              index = '[i]')
        eq = translator.parse()
        
        if post_target: # the left side has to be modified
            eq = eq.replace( "g_" + proj_desc['target'] + "_[i]",
                        "post_population_->g_"+proj_desc['target']+"_new_[post_neuron_rank_]")

        # Append the result of analysis
        pre_spike_var.append( { 'name': name, 'eq': eq } )

    return pre_spike_var 

def _extract_post_spike_variable(proj_desc):
    post_spike_var = []
    
    for var in _prepare_string(proj_desc['raw_post_spike']):
        name = _extract_name(var)
        translator = Equation(name, var, 
                              proj_desc['attributes'], 
                              proj_desc['local'], 
                              proj_desc['global'])
        eq = translator.parse()            
        post_spike_var.append( { 'name': name, 'eq': eq } )

    return post_spike_var    

####################################
# Functions for string manipulation
####################################
        
def _split_equation(definition):
    " Splits a description into equation and flags."
    try:
        equation, constraint = definition.rsplit(':', 1)
    except ValueError:
        equation = definition.strip() # there are no constraints
        constraint = None
    else:
        has_constraint = False
        for keyword in authorized_keywords:
            if keyword in constraint:
                has_constraint = True
        if has_constraint:
            equation = equation.strip()
            constraint = constraint.strip()
        else:
            equation = definition.strip() # there are no constraints
            constraint = None            
    finally:
        return equation, constraint
    
def _prepare_string(stream):
    """ Splits up a multiline equation, remove comments and unneeded spaces or tabs."""
    expr_set = []        
    # replace the ,,, by empty space and split the result up
    tmp_set = re.sub('\s+\.\.\.\s+', ' ', stream).split('\n')
    for expr in tmp_set:
        expr = re.sub('\#[\s\S]+', ' ', expr)   # remove comments
        expr = re.sub('\s+', ' ', expr)     # remove additional tabs etc.
        if expr.strip() == '' or len(expr)==0: # through beginning line breaks or something similar empty strings are contained in the set
            continue           
        expr_set.append(''.join(expr))        
    return expr_set 

def _extract_name(equation, left=False):
    " Extracts the name of a parameter/variable by looking the left term of an equation."
    equation = equation.replace(' ','')
    if not left: # there is potentially an equal sign
        try:
            name = equation.split('=')[0]
        except: # No equal sign. Eg: baseline : init=0.0
            return equation.strip()
        
        # Search for increments
        operators = ['+=', '-=', '*=', '/=', '>=', '<=']
        for op in operators:
            if op in equation: 
                return equation.split(op)[0]  
    else:
        name = equation.strip()          
        # Search for increments
        operators = ['+', '-', '*', '/']
        for op in operators:
            if equation.endswith(op): 
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
    name = re.findall("d([\w]+)/dt", name)
    if len(name) == 1:
        return name[0].strip()
    else:
        return '_undefined' 

def _extract_condition_name(equation):
    " Extracts the name of a parameter/variable by looking the left term of an equation."
    equation = equation.replace(' ','')
    # Search for operators
    operators = ['>', '<']
    for op in operators:
        if op in equation: 
            return equation.split(op)[0]  
    return '_undefined'   
    
                
def _extract_flags(constraint):
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
        
def _process_equations(equations):
    """ Takes a multi-string describing equations and returns a list of dictionaries, where:
    
    * 'name' is the name of the variable
    
    * 'eq' is the equation
    
    * 'constraints' is all the constraints given after the last :. _extract_flags() should be called on it.
    
    Warning: one equation can now be on multiple lines, without needing the ... newline symbol.
    
    TODO: should this be used for other arguments as equations? pre_event and so on
    """
    def is_constraint(eq):
        " Internal method to determine if a string contains reserved keywords." 
        eq = ',' +  eq.replace(' ', '') + ','
        for key in authorized_keywords:
            pattern = '([,]+)' + key + '([=,]+)'
            if re.match(pattern, eq):
                return True
        return False
    # All equations will be stored there, in the order of their definition
    variables = []        
    
    # Iterate over all lines
    for line in equations.splitlines():
        # Skip empty lines
        definition = line.strip()
        if definition == '':
            continue
        # Remove comments
        com = definition.split('#')
        if len(com) > 1:
            definition = com[0]
            if definition.strip() == '':
                continue
        # Process the line
        try:
            equation, constraint = definition.rsplit(':', 1)
        except ValueError: # There is no :, only equation is concerned 
            equation = line
            constraint = ''
        else:   # there is a :
            # Check if the constraint contains the reserved keywords
            has_constraint = is_constraint(constraint)
            # If the right part of : is a constraint, just store it
            # Otherwise, it is an if-then-else statement
            if has_constraint:
                equation = equation.strip()
                constraint = constraint.strip()
            else:
                equation = definition.strip() # there are no constraints
                constraint = '' 
        # Split the equation around operators = += -= *= /=, but not ==
        split_operators = re.findall('([\s\w\+\-\*\/]+)=([^=])', equation)
        if len(split_operators) == 1: # definition of a new variable
            # Retrieve the name
            name = _extract_name(split_operators[0][0], left=True)
            if name == '_undefined':
                _error('No variable name can be found in ' + equation)
                return []
            # Append the result
            variables.append({'name': name, 'eq': equation.strip(), 'constraint': constraint.strip()})
        elif len(split_operators) == 0: 
            # Continuation of the equation on a new line: append the equation to the previous variable
            variables[-1]['eq'] += ' ' + equation.strip()
            variables[-1]['constraint'] += constraint
        else:
            print 'Error: only one assignement operator is allowed per equation.'
            return []
    return variables  
