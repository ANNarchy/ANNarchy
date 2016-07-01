"""

    SingleAnalysis.py

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
from .CoupledEquations import CoupledEquations

def analyse_neuron(neuron):
    """ Performs the initial analysis for a single neuron type."""
    concurrent_odes = []

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

    # Make sure r is defined for rate-coded networks
    if neuron.type == 'rate':
        for var in description['parameters'] + description['variables']:
            if var['name'] == 'r':
                break
        else:
            _error('Rate-coded neurons must define the variable "r".')

    else: # spiking neurons define r by default, it contains the average FR if enabled
        for var in description['parameters'] + description['variables']:
            if var['name'] == 'r':
                _error('Spiking neurons use the variable "r" for the average FR, use another name.')

        description['variables'].append(
                {'name': 'r', 'locality': 'local', 'bounds': {}, 'ctype': 'double', 'init': 0.0,
                 'flags': [], 'eq': '', 'cpp': ""})

    # Extract functions
    functions = extract_functions(neuron.functions, False)
    description['functions'] = functions

    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error('Attributes must be declared only once.', attributes)

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
                    { 'name': 'g_'+target, 'locality': 'local', 'bounds': {}, 'ctype': 'double',
                        'init': 0.0, 'flags': [], 'eq': 'g_' + target+ ' = 0.0'}
                )
                description['attributes'].append('g_'+target)
                description['local'].append('g_'+target)

    # Extract RandomDistribution objects
    random_distributions = extract_randomdist(description)
    description['random_distributions'] = random_distributions

    # Extract the spike condition if any
    if neuron.type == 'spike':
        description['spike'] = extract_spike_variable(description)

    # Global operation TODO
    description['global_operations'] = []

    # Translate the equations to C++
    for variable in description['variables']:
        eq = variable['transformed_eq']
        if eq.strip() == "":
            continue
        untouched={}

        # Replace sum(target) with pop%(id)s.sum_exc[i]
        for target in description['targets']:
            eq = re.sub('sum\(\s*'+target+'\s*\)', '__sum_'+target+'__', eq)
            untouched['__sum_'+target+'__'] = '_sum_' + target + '%(local_index)s'

        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_neuron(variable['name'], eq, description)

        # Add the untouched variables to the global list
        for name, val in untouched_globs.items():
            if not name in untouched.keys():
                untouched[name] = val
        description['global_operations'] += global_ops

        # Extract if-then-else statements
        eq, condition = extract_ite(variable['name'], eq, description)

        # Find the numerical method if any
        method = find_method(variable)

        # Process the bounds
        if 'min' in variable['bounds'].keys():
            if isinstance(variable['bounds']['min'], str):
                translator = Equation(
                                variable['name'],
                                variable['bounds']['min'],
                                description,
                                type = 'return',
                                untouched = untouched
                                )
                variable['bounds']['min'] = translator.parse().replace(';', '')

        if 'max' in variable['bounds'].keys():
            if isinstance(variable['bounds']['max'], str):
                translator = Equation(
                                variable['name'],
                                variable['bounds']['max'],
                                description,
                                type = 'return',
                                untouched = untouched)
                variable['bounds']['max'] = translator.parse().replace(';', '')

        # Analyse the equation
        if condition == []:
            translator = Equation(
                                    variable['name'],
                                    eq,
                                    description,
                                    method = method,
                                    untouched = untouched
                            )
            code = translator.parse()
            dependencies = translator.dependencies()
        else: # An if-then-else statement
            code = translate_ITE(
                        variable['name'],
                        eq,
                        condition,
                        description,
                        untouched )
            dependencies = []


        if isinstance(code, str):
            cpp_eq = code
            switch = None
        else: # ODE
            cpp_eq = code[0]
            switch = code[1]

        # Replace untouched variables with their original name
        for prev, new in untouched.items():
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
            cpp_eq = re.sub(r'([^\w]*)'+f['name']+'\(', r'\1'+f['name'] + '(', ' ' + cpp_eq).strip()

        # Store the result
        variable['cpp'] = cpp_eq # the C++ equation
        variable['switch'] = switch # switch value of ODE
        variable['untouched'] = untouched # may be needed later
        variable['method'] = method # may be needed later
        variable['dependencies'] = dependencies # may be needed later

        # If the method is implicit or midpoint, the equations must be solved concurrently (depend on v[t+1])
        if method in ['implicit', 'midpoint']:
            concurrent_odes.append(variable)

    # After all variables are processed, do it again if they are concurrent
    if len(concurrent_odes) > 1 :
        solver = CoupledEquations(description, concurrent_odes)
        new_eqs = solver.process_variables()
        for idx, variable in enumerate(description['variables']):
            for new_eq in new_eqs:
                if variable['name'] == new_eq['name']:
                    description['variables'][idx] = new_eq

    return description

def analyse_synapse(synapse):
    """ Performs the analysis for a single synapse."""
    concurrent_odes = []

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
    elif synapse.type == 'rate':
        description['raw_psp'] = "w*pre.r"

    if synapse.type == 'spike': # Additionally store pre_spike and post_spike
        description['raw_pre_spike'] = synapse.pre_spike
        description['raw_post_spike'] = synapse.post_spike

    # Extract parameters and variables names
    parameters = extract_parameters(synapse.parameters, synapse.extra_values)
    variables = extract_variables(synapse.equations)

    # Extract functions
    functions = extract_functions(synapse.functions, False)

    # Check the presence of w
    description['plasticity'] = False
    for var in parameters + variables:
        if var['name'] == 'w':
            break
    else:
        parameters.append(
            #TODO: is this exception really needed? Maybe we could find
            #      a better solution instead of the hard-coded 'w' ... [hdin: 26.05.2015]
            {'name': 'w', 'bounds': {}, 'ctype': 'double', 'init': 0.0, 'flags': [], 'eq': 'w=0.0', 'locality': 'local'}
        )

    # Find out a plasticity rule
    for var in variables:
        if var['name'] == 'w':
            description['plasticity'] = True
            break

    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)

    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error('Attributes must be declared only once.', attributes)


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
    description['random_distributions'] = extract_randomdist(description)

    # Extract event-driven info
    if description['type'] == 'spike':
        # pre_spike event
        description['pre_spike'] = extract_pre_spike_variable(description)
        for var in description['pre_spike']:
            if var['name'] in ['g_target']: # Already dealt with
                continue
            for avar in description['variables']:
                if var['name'] == avar['name']:
                    break
            else: # not defined already
                description['variables'].append(
                {'name': var['name'], 'bounds': var['bounds'], 'ctype': var['ctype'], 'init': var['init'],
                 'locality': var['locality'],
                 'flags': [], 'transformed_eq': '', 'eq': '',
                 'cpp': '', 'switch': '', 'untouched': '', 'method':'explicit'}
                )
                description['local'].append(var['name'])
                description['attributes'].append(var['name'])

        # post_spike event
        description['post_spike'] = extract_post_spike_variable(description)
        for var in description['post_spike']:
            if var['name'] in ['g_target', 'w']: # Already dealt with
                continue
            for avar in description['variables']:
                if var['name'] == avar['name']:
                    break
            else: # not defined already
                description['variables'].append(
                {'name': var['name'], 'bounds': var['bounds'], 'ctype': var['ctype'], 'init': var['init'],
                 'locality': var['locality'],
                 'flags': [], 'transformed_eq': '', 'eq': '',
                 'cpp': '', 'switch': '', 'untouched': '', 'method':'explicit'}
                )
                description['local'].append(var['name'])
                description['attributes'].append(var['name'])

    # Variables names for the parser which should be left untouched
    untouched = {}
    description['dependencies'] = {'pre': [], 'post': []}

    # Iterate over all variables
    for variable in description['variables']:
        eq = variable['transformed_eq']

        # Event-driven variables
        if eq.strip() == '':
            continue

        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_synapse(variable['name'], eq, description)
        description['pre_global_operations'] += global_ops['pre']
        description['post_global_operations'] += global_ops['post']

        # Extract pre- and post_synaptic variables
        eq, untouched_var, dependencies = extract_prepost(variable['name'], eq, description)

        description['dependencies']['pre'] += dependencies['pre']
        description['dependencies']['post'] += dependencies['post']

        # Extract if-then-else statements
        eq, condition = extract_ite(variable['name'], eq, description)

        # Add the untouched variables to the global list
        for name, val in untouched_globs.items():
            if not name in untouched.keys():
                untouched[name] = val
        for name, val in untouched_var.items():
            if not name in untouched.keys():
                untouched[name] = val

        # Save the tranformed equation
        variable['transformed_eq'] = eq

        # Find the numerical method if any
        method = find_method(variable)

        # Process the bounds
        if 'min' in variable['bounds'].keys():
            if isinstance(variable['bounds']['min'], str):
                translator = Equation(variable['name'], variable['bounds']['min'],
                                      description,
                                      type = 'return',
                                      untouched = untouched.keys())
                variable['bounds']['min'] = translator.parse().replace(';', '')

        if 'max' in variable['bounds'].keys():
            if isinstance(variable['bounds']['max'], str):
                translator = Equation(variable['name'], variable['bounds']['max'],
                                      description,
                                      type = 'return',
                                      untouched = untouched.keys())
                variable['bounds']['max'] = translator.parse().replace(';', '')

        # Analyse the equation
        if condition == []: # Call Equation
            translator = Equation(variable['name'], eq,
                                  description,
                                  method = method,
                                  untouched = untouched.keys())
            code = translator.parse()
            dependencies = translator.dependencies()

        else: # An if-then-else statement
            code = translate_ITE(variable['name'], eq, condition, description,
                    untouched)
            dependencies = []

        if isinstance(code, str):
            cpp_eq = code
            switch = None
        else: # ODE
            cpp_eq = code[0]
            switch = code[1]

        # Replace untouched variables with their original name
        # print cpp_eq
        for prev, new in untouched.items():
            cpp_eq = cpp_eq.replace(prev, new)
        # print cpp_eq

        # Replace local functions
        for f in description['functions']:
            cpp_eq = re.sub(r'([^\w]*)'+f['name']+'\(', r'\1'+ f['name'] + '(', ' ' + cpp_eq).strip()

        # Store the result
        variable['cpp'] = cpp_eq # the C++ equation
        variable['switch'] = switch # switch value id ODE
        variable['untouched'] = untouched # may be needed later
        variable['method'] = method # may be needed later
        variable['dependencies'] = dependencies # may be needed later

        # If the method is implicit or midpoint, the equations must be solved concurrently (depend on v[t+1])
        if method in ['implicit', 'midpoint']:
            concurrent_odes.append(variable)

    # After all variables are processed, do it again if they are concurrent
    if len(concurrent_odes) > 1 :
        solver = CoupledEquations(description, concurrent_odes)
        new_eqs = solver.process_variables()
        for idx, variable in enumerate(description['variables']):
            for new_eq in new_eqs:
                if variable['name'] == new_eq['name']:
                    description['variables'][idx] = new_eq

    # Translate the psp code if any
    if 'raw_psp' in description.keys():
        psp = {'eq' : description['raw_psp'].strip() }
        # Extract global operations
        eq, untouched_globs, global_ops = extract_globalops_synapse('psp', " " + psp['eq'] + " ", description)
        description['pre_global_operations'] += global_ops['pre']
        description['post_global_operations'] += global_ops['post']
        # Replace pre- and post_synaptic variables
        eq, untouched, dependencies = extract_prepost('psp', eq, description)
        description['dependencies']['pre'] += dependencies['pre']
        description['dependencies']['post'] += dependencies['post']
        for name, val in untouched_globs.items():
            if not name in untouched.keys():
                untouched[name] = val
        # Extract if-then-else statements
        eq, condition = extract_ite('psp', eq, description, split=False)
        # Analyse the equation
        if condition == []:
            translator = Equation('psp', eq,
                                  description,
                                  method = 'explicit',
                                  untouched = untouched.keys(),
                                  type='return')
            code = translator.parse()
        else:
            code = translate_ITE('psp', eq, condition, description, untouched,
                                  split=False)

        # Replace untouched variables with their original name
        for prev, new in untouched.items():
            code = code.replace(prev, new)

        # Store the result
        psp['cpp'] = code
        description['psp'] = psp

    # Process event-driven info
    if description['type'] == 'spike':
        for variable in description['pre_spike'] + description['post_spike']:
            # Find plasticity
            if variable['name'] == 'w':
                description['plasticity'] = True
            # Retrieve the equation
            eq = variable['eq']
            # Extract if-then-else statements
            eq, condition = extract_ite(variable['name'], eq, description)
            # Analyse the equation
            if condition == []:
                translator = Equation(variable['name'], eq,
                                      description,
                                      method = 'explicit',
                                      untouched = {})
                code = translator.parse()
            else:
                code = translate_ITE(variable['name'], eq, condition, description, {}, split=True)
            # Store the result
            variable['cpp'] = code # the C++ equation

            # Process the bounds
            if 'min' in variable['bounds'].keys():
                if isinstance(variable['bounds']['min'], str):
                    translator = Equation(
                                    variable['name'],
                                    variable['bounds']['min'],
                                    description,
                                    type = 'return',
                                    untouched = untouched )
                    variable['bounds']['min'] = translator.parse().replace(';', '')

            if 'max' in variable['bounds'].keys():
                if isinstance(variable['bounds']['max'], str):
                    translator = Equation(
                                    variable['name'],
                                    variable['bounds']['max'],
                                    description,
                                    type = 'return',
                                    untouched = untouched)
                    variable['bounds']['max'] = translator.parse().replace(';', '')

    # Structural plasticity
    if synapse.pruning:
        description['pruning'] = extract_structural_plasticity(synapse.pruning, description)
    if synapse.creating:
        description['creating'] = extract_structural_plasticity(synapse.creating, description)

    return description
