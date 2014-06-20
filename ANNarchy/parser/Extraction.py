"""

    Extraction.py

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
from ANNarchy.core.Random import available_distributions, distributions_arguments, distributions_templates
from ANNarchy.parser.Equation import Equation
from ANNarchy.parser.Function import FunctionParser
from ANNarchy.parser.StringManipulation import *
from ANNarchy.parser.ITE import *

import re

def extract_randomdist(pop):
    " Extracts RandomDistribution objects from all variables"
    rk_rand = 0
    random_objects = []
    for variable in pop.description['variables']:
        eq = variable['eq']
        # Search for all distributions
        for dist in available_distributions:
            matches = re.findall('(?P<pre>[^\_a-zA-Z0-9.])'+dist+'\(([^()]+)\)', eq)
            if matches == ' ':
                continue
            for l, v in matches:
                # Check the arguments
                arguments = v.split(',')
                # Check the number of provided arguments
                if len(arguments) < distributions_arguments[dist]:
                    _error(eq)
                    _error('The distribution ' + dist + ' requires ' + str(distributions_arguments[dist]) + 'parameters')
                elif len(arguments) == distributions_arguments[dist]:
                    arguments.append(str(config['seed']))
                elif len(arguments) == distributions_arguments[dist] +1 :
                    _warning('The seed is set in the distribution ' + dist)
                else:
                    _error(eq)
                    _error('Too many parameters provided to the distribution ' + dist)
                # Process the arguments
                processed_arguments = ""
                for idx in range(len(arguments)):
                    try:
                        arg = float(arguments[idx])
                    except: # A global parameter
                        if arguments[idx].strip() in pop.description['global']:
                            arg = arguments[idx] + '_'
                        else:
                            _error(arguments[idx] + ' is not a global parameter of the neuron/synapse. It can not be used as an argument to the random distribution ' + dist + '(' + v + ')')
                            exit(0)

                    processed_arguments += str(arg)
                    if idx != len(arguments)-1: # not the last one
                        processed_arguments += ', '
                definition = dist + '(' + processed_arguments + ')'
                # Store its definition
                desc = {'name': '__rand_' + str(rk_rand) + '_',
                        'dist': dist,
                        'definition': definition,
                        'args' : processed_arguments,
                        'template': distributions_templates[dist]}
                rk_rand += 1
                random_objects.append(desc)
                # Replace its definition by its temporary name
                # Problem: when one uses twice the same RD in a single equation (perverse...)
                eq = eq.replace(dist+'('+v+')', desc['name'])
                # Add the new variable to the vocabulary
                pop.description['attributes'].append(desc['name'])
                if variable['name'] in pop.description['local']:
                    pop.description['local'].append(desc['name'])
                else: # Why not on a population-wide variable?
                    pop.description['global'].append(desc['name'])
        variable['transformed_eq'] = eq
        
    return random_objects
    
def extract_globalops_neuron(name, eq, pop):
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
    
def extract_globalops_synapse(name, eq, proj):
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
        # if len(pre_matches) > 0 and name in proj.description['global']:
        #     _error(eq + '\nA postsynaptic variable can not depend on pre.' + pre_matches[0])
        #     exit(0)
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
    
def extract_prepost(name, eq, proj):
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
                   

def extract_parameters(description, extra_values):
    """ Extracts all variable information from a multiline description."""
    parameters = []
    # Split the multilines into individual lines
    parameter_list = prepare_string(description)
    
    # Analyse all variables
    for definition in parameter_list:
        # Check if there are flags after the : symbol
        equation, constraint = split_equation(definition)
        # Extract the name of the variable
        name = extract_name(equation)
        if name == '_undefined':
            exit(0)
        # Process the flags if any
        bounds, flags = extract_flags(constraint)

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
    
def extract_variables(description):
    """ Extracts all variable information from a multiline description."""
    variables = []
    # Split the multilines into individual lines
    variable_list = process_equations(description)
    # Analyse all variables
    for definition in variable_list:
        # Retrieve the name, equation and constraints for the variable
        equation = definition['eq']
        constraint = definition['constraint']
        name = definition['name']
        if name == '_undefined':
            exit(0)
        # Process the flags if any
        bounds, flags = extract_flags(constraint)
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
    
def extract_functions(description, local_global=False):
    """ Extracts all functions from a multiline description."""
    if not description:
        return [] 
    # Split the multilines into individual lines
    function_list = process_equations(description)
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
        eq2, condition = extract_ite('', content, None, split=False)
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
    
    
def get_attributes(parameters, variables):
    """ Returns a list of all attributes names, plus the lists of local/global variables."""
    attributes = []; local_var = []; global_var = []
    for p in parameters + variables:
        attributes.append(p['name'])
        if 'population' in p['flags'] or 'postsynaptic' in p['flags']:
            global_var.append(p['name'])
        else:
            local_var.append(p['name'])
    return attributes, local_var, global_var

def extract_targets(variables):
    targets = []
    for var in variables:
        # Rate-coded neurons
        code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])sum\(([^()]+)\)', var['eq'])
        for l, t in code:
            targets.append(t)
        # Spiking neurons
        code = re.findall('([^\_a-zA-Z0-9.])g_([\w]+)', var['eq'])
        for l, t in code:
            targets.append(t)
    return list(set(targets))

def extract_spike_variable(pop_desc):
    spike_name = extract_condition_name(pop_desc['raw_spike'].strip())
    translator = Equation('raw_spike_cond', pop_desc['raw_spike'], 
                          pop_desc['attributes'], 
                          pop_desc['local'], 
                          pop_desc['global'], 
                          type = 'cond')
    raw_spike_code = translator.parse()
    
    raw_reset_code = ''
    if pop_desc.has_key('raw_reset') and pop_desc['raw_reset']:
        for var in prepare_string(pop_desc['raw_reset']):
            name = extract_name(var)
            translator = Equation(name, var, 
                                  pop_desc['attributes'], 
                                  pop_desc['local'], 
                                  pop_desc['global'])
            raw_reset_code += translator.parse() +'\n'
    
    return { 'name': spike_name, 'spike_cond': raw_spike_code, 'spike_reset': raw_reset_code}

def extract_pre_spike_variable(proj):
    pre_spike_var = []
    # For all variables influenced by a presynaptic spike
    for var in prepare_string(proj.description['raw_pre_spike']):
        # Get its name
        name = extract_name(var)
        raw_eq = var
        # Replace target by its value
        post_target = False
        if name == "g_target":
            name = name.replace("target", proj.description['target'])
            post_target = True

        # Extract if-then-else statements
        eq, condition = extract_ite(name, var, proj)
            
        if condition == []:
            translator = Equation(name, var, 
                                  proj.description['attributes'] + [name], 
                                  proj.description['local'] + [name], 
                                  proj.description['global'],
                                  index = '[i]')
            eq = translator.parse()
        else: 
            eq = translate_ITE(name, eq, condition, proj, {})
        
        if post_target: # the left side has to be modified
            eq = eq.replace( "g_" + proj.description['target'] + "_[i]",
                        "post_population_->g_"+proj.description['target']+"_new_[post_neuron_rank_]")

        # Append the result of analysis
        pre_spike_var.append( { 'name': name, 'eq': eq , 'raw_eq' : raw_eq} )

    return pre_spike_var 

def extract_post_spike_variable(proj):
    post_spike_var = []
    
    for var in prepare_string(proj.description['raw_post_spike']):
        name = extract_name(var)

        # Extract if-then-else statements
        eq, condition = extract_ite(name, var, proj)

        if condition == []:
            translator = Equation(name, var, 
                                  proj.description['attributes'], 
                                  proj.description['local'], 
                                  proj.description['global'])
            eq = translator.parse()     
        else: 
            eq = translate_ITE(name, eq, condition, proj, {}) 

        post_spike_var.append( { 'name': name, 'eq': eq, 'raw_eq' : var} )

    return post_spike_var  

def extract_stop_condition(pop):
    eq = pop['stop_condition']['eq']
    pop['stop_condition']['type'] = 'any'
    # Check the flags
    split = eq.split(':')
    if len(split) > 1: # flag given
        eq = split[0]
        flags = split[1].strip()
        split = flags.split(' ')
        for el in split:
            if el.strip() == 'all':
                pop['stop_condition']['type'] = 'all'
    # Convert the expression
    translator = Equation('stop_cond', eq, 
                          pop['attributes'], 
                          pop['local'], 
                          pop['global'], 
                          type = 'cond')
    code = translator.parse()
    pop['stop_condition']['cpp'] = '(' + code + ')'