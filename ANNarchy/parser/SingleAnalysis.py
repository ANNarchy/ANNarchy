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
from .Analyser import *
from ANNarchy.core.Global import _error, _warning

def analyse_population(pop):
    """ Performs the initial analysis for a single population."""
    # Identify the population type
    pop_type = pop.neuron_type.type
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
    parameters = extract_parameters(pop.neuron_type.parameters, pop.neuron_type.extra_values)
    variables = extract_variables(pop.neuron_type.equations)
    # Extract functions
    functions = extract_functions(pop.neuron_type.functions, False)
    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)
    # Test if attributes are declared only once
    if len(attributes) != len(list(set(attributes))):
        _error(pop.name, ': attributes must be declared only once.', attributes)
        exit(0)
    # Extract the stop condition
    if pop._stop_condition:
        description['stop_condition'] = {'eq': pop._stop_condition}
    
    # Extract all targets
    targets = extract_targets(variables)

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

    # Store basic information
    description = {
        'pre': proj.pre.name,
        'pre_class': proj.pre.class_name,
        'post': proj.post.name,
        'post_class': proj.post.class_name,
        'target': proj.target,
        'type': proj.type,
        'raw_parameters': proj.synapse_type.parameters,
        'raw_equations': proj.synapse_type.equations,
        'raw_functions': proj.synapse_type.functions
    }

    if proj.type == 'spike': # Additionally store pre_spike and post_spike
        if proj.synapse_type.pre_spike:
            description['raw_pre_spike'] = proj.synapse_type.pre_spike
        else: # pre_spike not defined, but other fields yes
            description['raw_pre_spike'] = "g_target += w"
        description['raw_post_spike'] = proj.synapse_type.post_spike
    else:
        if proj.synapse_type.psp:
            description['raw_psp'] = proj.synapse_type.psp

    # Extract parameters and variables names
    parameters = extract_parameters(proj.synapse_type.parameters, proj.synapse_type.extra_values)
    variables = extract_variables(proj.synapse_type.equations)
    # Extract functions
    functions = extract_functions(proj.synapse_type.functions, False)
    # Build lists of all attributes (param+var), which are local or global
    attributes, local_var, global_var = get_attributes(parameters, variables)
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