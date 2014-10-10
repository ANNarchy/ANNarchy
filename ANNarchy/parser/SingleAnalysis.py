"""

    SSingleAnalysis.py

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
from ANNarchy.core.Global import _error, _warning, config
from .Extraction import *

# Specific code generation for the chosen paradigm
pattern_omp = {
    # Populations
    'pop_prefix': 'pop%(id)s',
    'pop_sep': '.',
    'pop_index': '[i]',
    'pop_globalindex': '',
    'pop_sum': '_sum_',
    # Projections
    'proj_prefix': 'proj%(id_proj)s',
    'proj_sep': '.',
    'proj_index': '[i][j]',
    'proj_globalindex': '[i]',
    'proj_preprefix': 'pop%(id_pre)s',
    'proj_postprefix': 'pop%(id_post)s',
    'proj_preindex': '[rk_pre]',
    'proj_postindex': '[rk_post]',
}
pattern_cuda = {
    # Populations
    'pop_prefix': 'pop%(id)s',
    'pop_sep': '.',
    'pop_index': '[i]',
    'pop_globalindex': '',
    'pop_sum': 'sum_',
    # Projections
    'proj_prefix': 'proj%(id_proj)s',
    'proj_sep': '.',
    'proj_index': '[i][j]',
    'proj_globalindex': '[i]',
    'proj_preprefix': 'pop%(id_pre)s',
    'proj_postprefix': 'pop%(id_post)s',
}

def analyse_neuron(neuron):
    """ Performs the initial analysis for a single neuron type."""

    # Find the paradigm OMP or CUDA
    if config['paradigm'] == 'cuda':
        pattern = pattern_cuda
    else:
        pattern = pattern_omp

    # Store basic information
    description = {
        'object': 'neuron',
        'type': neuron.type,
        'raw_parameters': neuron.parameters,
        'raw_equations': neuron.equations,
        'raw_functions': neuron.functions,
    }

    if neuron.type == 'spike': # Additionally store reset and spike
        description['raw_reset'] = neuron.reset
        description['raw_spike'] = neuron.spike
        description['refractory'] = neuron.refractory
        
    # Extract parameters and variables names
    parameters = extract_parameters(neuron.parameters, neuron.extra_values)
    variables = extract_variables(neuron.equations)
    description['parameters'] = parameters
    description['variables'] = variables

    # Extract functions
    functions = extract_functions(neuron.functions, False)
    description['functions'] = functions

    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error('Attributes must be declared only once.', attributes)
        exit(0)
    description['attributes'] = attributes
    description['local'] = local_var
    description['global'] = global_var
    
    # Extract all targets
    targets = extract_targets(variables)
    description['targets'] = targets
    if neuron.type == 'spike': # Add a default reset behaviour for conductances
        for target in targets:
            found = False
            for var in description['variables']:
                if var['name'] == 'g_' + target:
                    found = True
                    break
            if not found:
                description['variables'].append(
                    {'name': 'g_'+target, 'bounds': {}, 'ctype': 'double', 
                        'init': 0.0, 'flags': [], 'eq': 'g_' + target+ ' = 0.0'}
                )
                description['attributes'].append('g_'+target)
                description['local'].append('g_'+target)

    # Extract RandomDistribution objects
    random_distributions = extract_randomdist(description, pattern)
    description['random_distributions'] = random_distributions

    # Extract the spike condition if any
    if neuron.type == 'spike':
        description['spike'] = extract_spike_variable(description, pattern)

    # Global operation TODO
    description['global_operations'] = []

    # Translate the equations to C++
    for variable in description['variables']:
        eq = variable['transformed_eq']
        untouched={}
        
        # Replace sum(target) with pop%(id)s.sum_exc[i]
        for target in description['targets']:
            eq = eq.replace('sum('+target+')', '__sum_'+target+'__' )  
            untouched['__sum_'+target+'__'] = pattern['pop_prefix'] + pattern['pop_sep'] + pattern['pop_sum'] + target + pattern['pop_index']
        
        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_neuron(variable['name'], eq, description, pattern)

        # Add the untouched variables to the global list
        for name, val in untouched_globs.iteritems():
            if not untouched.has_key(name):
                untouched[name] = val
        description['global_operations'] += global_ops
        
        # Extract if-then-else statements
        eq, condition = extract_ite(variable['name'], eq, description)
        
        # Find the numerical method if any
        method = find_method(variable) 

        # Process the bounds
        if 'min' in variable['bounds'].keys():
            if isinstance(variable['bounds']['min'], str):
                translator = Equation(variable['name'], variable['bounds']['min'], 
                                      description['attributes'], 
                                      description['local'], 
                                      description['global'], 
                                      type = 'return',
                                      untouched = untouched.keys(),
                                      prefix=pattern['pop_prefix'],
                                      sep=pattern['pop_sep'],
                                      index=pattern['pop_index'],
                                      global_index=pattern['pop_globalindex'],
                                      )
                variable['bounds']['min'] = translator.parse().replace(';', '')

        if 'max' in variable['bounds'].keys():
            if isinstance(variable['bounds']['max'], str):
                translator = Equation(variable['name'], variable['bounds']['max'], 
                                      description['attributes'], 
                                      description['local'], 
                                      description['global'], 
                                      type = 'return',
                                      untouched = untouched.keys(),
                                      prefix=pattern['pop_prefix'],
                                      sep=pattern['pop_sep'],
                                      index=pattern['pop_index'],
                                      global_index=pattern['pop_globalindex'],)
                variable['bounds']['max'] = translator.parse().replace(';', '')
        
        # Analyse the equation
        if condition == []:
            translator = Equation(variable['name'], eq, 
                                  description['attributes'], 
                                  description['local'], 
                                  description['global'], 
                                  method = method,
                                  untouched = untouched.keys(),
                                  prefix=pattern['pop_prefix'],
                                  sep=pattern['pop_sep'],
                                  index=pattern['pop_index'],
                                  global_index=pattern['pop_globalindex'],)
            code = translator.parse()
        else: # An if-then-else statement
            code = translate_ITE(variable['name'], eq, condition, description, untouched,
                                  prefix=pattern['pop_prefix'],
                                  sep=pattern['pop_sep'],
                                  index=pattern['pop_index'],
                                  global_index=pattern['pop_globalindex'])

        
        if isinstance(code, str):
            cpp_eq = code
            switch = None
        else: # ODE
            cpp_eq = code[0]
            switch = code[1]

        # Replace untouched variables with their original name
        for prev, new in untouched.iteritems():
            if prev.startswith('g_'):
                cpp_eq = re.sub(r'([^_]+)'+prev, r'\1'+new, ' ' + cpp_eq).strip()
                if switch:
                    switch = re.sub(r'([^_]+)'+prev, new, ' ' + switch).strip()

            else:
                cpp_eq = re.sub(prev, new, cpp_eq)
                if switch:
                    switch = re.sub(prev, new, switch)

        # Replace local functions
        for f in description['functions']:
            cpp_eq = re.sub(r'([^\w]*)'+f['name']+'\(', r'\1'+pattern['pop_prefix'] + pattern['pop_sep'] + f['name'] + '(', ' ' + cpp_eq).strip()

        # Store the result
        variable['cpp'] = cpp_eq # the C++ equation
        variable['switch'] = switch # switch value of ODE
        variable['untouched'] = untouched # may be needed later
        variable['method'] = method # may be needed later

    return description

def analyse_synapse(synapse):  
    """ Performs the analysis for a single synapse."""  
 
    # Find the paradigm OMP or CUDA
    if config['paradigm'] == 'cuda':
        pattern = pattern_cuda
    else:
        pattern = pattern_omp  

    # Store basic information
    description = {
        'object': 'synapse',
        'type': synapse.type,
        'raw_parameters': synapse.parameters,
        'raw_equations': synapse.equations,
        'raw_functions': synapse.functions
    }

    if synapse.psp:
        description['raw_psp'] = synapse.psp

    if synapse.type == 'spike': # Additionally store pre_spike and post_spike
        description['raw_pre_spike'] = synapse.pre_spike
        description['raw_post_spike'] = synapse.post_spike

    # Extract parameters and variables names
    parameters = extract_parameters(synapse.parameters, synapse.extra_values)
    variables = extract_variables(synapse.equations)

    # Extract functions
    functions = extract_functions(synapse.functions, False)

    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error('Attributes must be declared only once.', attributes)
        exit(0)
        
    # Add this info to the description
    description['parameters'] = parameters
    description['variables'] = variables
    description['functions'] = functions
    description['attributes'] = attributes
    description['local'] = local_var
    description['global'] = global_var
    description['global_operations'] = []

    description['pre_global_operations'] = []
    description['post_global_operations'] = []

    # Extract RandomDistribution objects
    description['random_distributions'] = extract_randomdist(description, pattern)
    
    # Extract event-driven info TODO: check
    if description['type'] == 'spike':         
            description['pre_spike'] = extract_pre_spike_variable(description, pattern)
            description['post_spike'] = extract_post_spike_variable(description, pattern)

    # Variables names for the parser which should be left untouched
    untouched = {}   
    description['dependencies'] = {'pre': [], 'post': []}
                   
    # Iterate over all variables
    for variable in description['variables']:
        eq = variable['transformed_eq']
        
        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_synapse(variable['name'], eq, description, pattern)
        description['pre_global_operations'] += global_ops['pre']
        description['post_global_operations'] += global_ops['post']
        
        # Extract pre- and post_synaptic variables
        eq, untouched_var, dependencies = extract_prepost(variable['name'], eq, description, pattern)

        description['dependencies']['pre'] += dependencies['pre']
        description['dependencies']['post'] += dependencies['post']
        
        # Extract if-then-else statements
        eq, condition = extract_ite(variable['name'], eq, description)
        
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
        method = find_method(variable)

        # Process the bounds
        if 'min' in variable['bounds'].keys():
            if isinstance(variable['bounds']['min'], str):
                translator = Equation(variable['name'], variable['bounds']['min'], 
                                      description['attributes'], 
                                      description['local'], 
                                      description['global'], 
                                      type = 'return',
                                      untouched = untouched.keys(),
                                      prefix=pattern['proj_prefix'],
                                      sep=pattern['proj_sep'],
                                      index=pattern['proj_index'],
                                      global_index=pattern['proj_globalindex'])
                variable['bounds']['min'] = translator.parse().replace(';', '')

        if 'max' in variable['bounds'].keys():
            if isinstance(variable['bounds']['max'], str):
                translator = Equation(variable['name'], variable['bounds']['max'], 
                                      description['attributes'], 
                                      description['local'], 
                                      description['global'], 
                                      type = 'return',
                                      untouched = untouched.keys(),
                                      prefix=pattern['proj_prefix'],
                                      sep=pattern['proj_sep'],
                                      index=pattern['proj_index'],
                                      global_index=pattern['proj_globalindex'])
                variable['bounds']['max'] = translator.parse().replace(';', '')
            
        # Analyse the equation
        if condition == []: # Call Equation
            translator = Equation(variable['name'], eq, 
                                  description['attributes'], 
                                  description['local'], 
                                  description['global'], 
                                  method = method, 
                                  untouched = untouched.keys(),
                                  prefix=pattern['proj_prefix'],
                                  sep=pattern['proj_sep'],
                                  index=pattern['proj_index'],
                                  global_index=pattern['proj_globalindex'])
            code = translator.parse()
                
        else: # An if-then-else statement
            code = translate_ITE(variable['name'], eq, condition, description, untouched,
                                  prefix=pattern['proj_prefix'],
                                  sep=pattern['proj_sep'],
                                  index=pattern['proj_index'],
                                  global_index=pattern['proj_globalindex'])

        if isinstance(code, str):
            cpp_eq = code
            switch = None
        else: # ODE
            cpp_eq = code[0]
            switch = code[1]

        # Replace untouched variables with their original name
        for prev, new in untouched.iteritems():
            cpp_eq = cpp_eq.replace(prev, new)


        # Replace local functions
        for f in description['functions']:
            cpp_eq = re.sub(r'([^\w]*)'+f['name']+'\(', r'\1'+pattern['proj_prefix'] + pattern['proj_sep'] + f['name'] + '(', ' ' + cpp_eq).strip()     
        
        # Store the result
        variable['cpp'] = cpp_eq # the C++ equation
        variable['switch'] = switch # switch value id ODE
        variable['untouched'] = untouched # may be needed later
        variable['method'] = method # may be needed later
        
    # Translate the psp code if any
    if 'raw_psp' in description.keys():                
        psp = {'eq' : description['raw_psp'].strip() }
        # Replace pre- and post_synaptic variables
        eq = psp['eq']
        eq, untouched, dependencies = extract_prepost('psp', eq, description, pattern)
        # Extract if-then-else statements
        eq, condition = extract_ite('psp', eq, description, split=False)
        # Analyse the equation
        if condition == []:
            translator = Equation('psp', eq, 
                                  description['attributes'], 
                                  description['local'], 
                                  description['global'], 
                                  method = 'explicit', 
                                  untouched = untouched.keys(),
                                  type='return',
                                  prefix=pattern['proj_prefix'],
                                  sep=pattern['proj_sep'],
                                  index=pattern['proj_index'],
                                  global_index=pattern['proj_globalindex'])
            code = translator.parse()
        else:
            code = translate_ITE('psp', eq, condition, synapse, untouched,
                                  prefix=pattern['proj_prefix'],
                                  sep=pattern['proj_sep'],
                                  index=pattern['proj_index'],
                                  global_index=pattern['proj_globalindex'], 
                                  split=False)

        # Replace untouched variables with their original name
        for prev, new in untouched.iteritems():
            code = code.replace(prev, new) 

        # Store the result
        psp['cpp'] = code
        description['psp'] = psp   

    return description     