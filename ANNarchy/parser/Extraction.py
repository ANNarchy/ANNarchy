"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Random import *
from ANNarchy.core.Parameters import Parameter, Variable, Creating, Pruning
from ANNarchy.parser.Equation import Equation
from ANNarchy.parser.Function import FunctionParser
from ANNarchy.parser.StringManipulation import *
from ANNarchy.parser.ITE import *
from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm, get_global_config
from ANNarchy.intern import Messages

import re


def extract_randomdist(description, net_id):
    " Extracts RandomDistribution objects from all variables"
    rk_rand = 0
    random_objects = []
    for variable in description['variables']:
        # Equation
        eq = variable['eq']
        # Dependencies
        dependencies = []
        # Search for all distributions
        for dist in available_distributions:
            matches = re.findall(r'(?P<pre>[^\w.])'+dist+r'\(([^()]+)\)', eq)
            if matches == ' ':
                continue
            for l, v in matches:

                # Check the arguments
                arguments = v.split(',')

                # Check the number of provided arguments
                if len(arguments) < distributions_arguments[dist]:
                    Messages._print(eq)
                    Messages._error('The distribution ' + dist + ' requires ' + str(distributions_arguments[dist]) + 'parameters')
                elif len(arguments) > distributions_arguments[dist]:
                    Messages._print(eq)
                    Messages._error('Too many parameters provided to the distribution ' + dist)

                # Process the arguments
                processed_arguments = ""
                for idx in range(len(arguments)):
                    try:
                        arg = float(arguments[idx])
                    except: # A global parameter
                        if arguments[idx].strip() in description['global']:
                            arg = arguments[idx].strip() + "%(global_index)s"
                            dependencies.append(arguments[idx].strip())
                        else:
                            Messages._error(arguments[idx] + ' is not a global parameter of the neuron/synapse. It can not be used as an argument to the random distribution ' + dist + '(' + v + ')')

                    processed_arguments += str(arg)
                    if idx != len(arguments)-1: # not the last one
                        processed_arguments += ', '

                definition = distributions_equivalents[dist] + '(' + processed_arguments + ')'

                # Store its definition
                desc = {
                    'name': 'rand_' + str(rk_rand),
                    'dist': dist,
                    'definition': definition,
                    'args': processed_arguments,
                    'template': distributions_equivalents[dist],
                    'locality': variable['locality'],
                    'ctype': ConfigManager().get('precision', net_id),
                    'dependencies': dependencies
                }
                rk_rand += 1
                random_objects.append(desc)

                # Replace its definition by its temporary name
                # Problem: when one uses twice the same RD in a single equation (perverse...)
                eq = eq.replace(dist+'('+v+')', desc['name'])
                # Add the new variable to the vocabulary
                description['attributes'].append(desc['name'])
                if variable['name'] in description['local']:
                    description['local'].append(desc['name'])
                elif variable['name'] in description['semiglobal']:
                    description['semiglobal'].append(desc['name'])
                else: # Why not on a population-wide variable?
                    description['global'].append(desc['name'])

        variable['transformed_eq'] = eq

    return random_objects

def extract_globalops_neuron(name, eq, description, net_id):
    """ Replaces global operations (mean(r), etc)  with arbitrary names and
    returns a dictionary of changes.
    """
    untouched = {}
    globs = []
    # Global ops
    glop_names = ['min', 'max', 'mean', 'norm1', 'norm2']
    for op in glop_names:
        matches = re.findall(r'([^\w]*)' + op + r'\(([\s\w]*)\)', eq)
        for pre, var in matches:
            if var.strip() in description['local']:
                globs.append({'function': op, 'variable': var.strip()})
                oldname = op + '(' + var + ')'
                newname = '_' + op + '_' + var.strip()
                eq = eq.replace(oldname, newname)
                untouched[newname] = '_' + op + '_' + var.strip()
            else:
                Messages._print(eq)
                Messages._error('There is no local attribute '+var+'.')

    return eq, untouched, globs

def extract_globalops_synapse(name, eq, desc, net_id):
    """
    Replaces global operations (mean(pre.r), etc)  with arbitrary names and
    returns a dictionary of changes.
    """
    untouched = {}
    globs = {'pre' : [],
             'post' : [] }
    glop_names = ['min', 'max', 'mean', 'norm1', 'norm2']
    origin_eq = eq  # temporary copy only for error message

    for op in glop_names:
        pre_matches = re.findall(r'([^\w.])' + op + r'\(\s*pre\.([\w]+)\s*\)', eq)
        post_matches = re.findall(r'([^\w.])' + op + r'\(\s*post\.([\w]+)\s*\)', eq)
        proj_matches = re.findall(r'([^\w.])' + op + r'\(\s*([\w]+)\s*\)', eq)

        for pre, var in pre_matches:
            globs['pre'].append({'function': op, 'variable': var.strip()})
            newname =  '__pre_' + op + '_' + var.strip()
            eq = re.sub(op+r'\(\s*pre\.([\w]+)\s*\)', newname, eq)
            untouched[newname] = '%(pre_prefix)s_' + op + '_' + var

        for pre, var in post_matches:
            globs['post'].append({'function': op, 'variable': var.strip()})
            newname = '__post_' + op + '_' + var.strip()
            eq = re.sub(op+r'\(\s*post\.([\w]+)\s*\)', newname, eq)
            untouched[newname] = '%(post_prefix)s_' + op + '_' + var

        for _, var in proj_matches:
            Messages._error("Detected global operation '"+op+"' on synaptic variable '"+var+"' in equation '"+origin_eq+"' which is not implemented.")

    return eq, untouched, globs

def extract_prepost(name, eq, description, net_id):
    " Replaces pre.var and post.var with arbitrary names and returns a dictionary of changes."

    dependencies = {'pre': [], 'post': []}

    pre_matches = re.findall(r'pre\.([\w]+)', eq)
    post_matches = re.findall(r'post\.([\w]+)', eq)

    untouched = {}
    # Replace all pre.* occurences with a temporary variable
    for var in list(set(pre_matches)):
        if var == 'sum': # pre.sum(exc)
            def idx_target(val):
                target = val.group(1).strip()
                if target == '':
                    Messages._print(eq)
                    Messages._error('pre.sum() requires one argument.')

                rep = '_pre_sum_' + target.strip()
                dependencies['pre'].append('sum('+target+')')
                untouched[rep] = '%(pre_prefix)s_sum_' +target+ '%(pre_index)s'
                return rep

            eq = re.sub(r'pre\.sum\(([\s\w]+)\)', idx_target, eq)
        else:
            dependencies['pre'].append(var)
            eq = re.sub(r"pre." + var + r"([^_\w]+)", "_pre_" + var + r"__\g<1>", eq + " ")
            # eq = eq.replace(target, ' _pre_'+var)
            untouched['_pre_'+var+'__'] = '%(pre_prefix)s' + var + '%(pre_index)s'

    # Replace all post.* occurences with a temporary variable
    for var in list(set(post_matches)):
        if var == 'sum': # post.sum(exc)
            def idx_target(val):
                target = val.group(1).strip()
                if target == '':
                    Messages._print(eq)
                    Messages._error('post.sum() requires one argument.')

                dependencies['post'].append('sum('+target+')')
                rep = '_post_sum_' + target.strip()
                untouched[rep] = '%(post_prefix)s_sum_' + target + '%(post_index)s'
                return rep
            eq = re.sub(r'post\.sum\(([\s\w]+)\)', idx_target, eq)
        else:
            dependencies['post'].append(var)
            eq = re.sub(r"post." + var + r"([^_\w]+)", r"_post_" + var + r"__\g<1>", eq + " ")
            # eq = eq.replace(target, ' _post_'+var+'__')
            untouched['_post_'+var+'__'] = '%(post_prefix)s' + var +'%(post_index)s'

    return eq, untouched, dependencies


def extract_parameters(description, extra_values={}, object_type="neuron", net_id=0):
    """
    Extracts all variable information from a multiline description or a dictionary.
    """
    result = []

    if isinstance(description, (dict,)):

        for key, val in description.items():
            
            # Locality is determined by localparam/globalparam. The default is global
            if isinstance(val, (Parameter,)):
                # Get fields
                value = val.value
                locality = val.locality
                if locality in ['population', 'projection']: # for people used to this
                    locality = 'global'
                if locality in ['postsynaptic']:
                    locality = 'semiglobal'
                if not locality in ['global', 'semiglobal', 'local']:
                    Messages._error(f"Parameter {key}: the locality must be in ['global', 'semiglobal', 'local'].")
                
                ctype = val.type
                # Possibility to give the built-in type instead of the string
                if ctype == float:
                    ctype = 'float'
                if ctype == int:
                    ctype = 'int'
                if ctype == bool:
                    ctype = 'bool'
                if not ctype in ['float', 'int', 'bool']:
                    Messages._error(f"Parameter {key}: the data type must be in ['float', 'int', 'bool'].")

                # Recreate the string flags for compatibility
                if locality == 'global':
                    flags = 'population' if object_type == 'neuron' else 'projection'
                elif locality == 'semiglobal':
                    flags = 'postsynaptic'
                else:
                    flags = ""
                if ctype in ['int', 'bool']:
                    flags += ", " + ctype
            else: 
                locality = 'global'
                value = val
                ctype = 'float'
                flags = 'population' if object_type == 'neuron' else 'projection'

            # ctype 
            ctype = ConfigManager().get('precision', net_id) if ctype == 'float' else ctype

            # Store the result
            desc = {'name': key,
                    'locality': locality,
                    'eq': key + " = " + str(value),
                    'bounds': {},
                    'flags' : flags,
                    'ctype' : ctype,
                    'init' : value,
                    }
            result.append(desc)
    
    elif isinstance(description, (str, list, )):
    
        # Split the multilines into individual lines
        if isinstance(description, (str,)):
            description = prepare_string(description)

        # Analyse all variables
        for definition in description:
            # Check if there are flags after the : symbol
            equation, constraint = split_equation(definition)
            
            # Extract the name of the variable
            name = extract_name(equation)
            if name in ['_undefined', ""]:
                Messages._error("Definition can not be analysed: " + equation)

            # Process constraint
            bounds, flags, ctype, init = extract_boundsflags(constraint, equation, extra_values, net_id=net_id)

            # Determine locality
            for f in ['population', 'postsynaptic', 'projection']:
                if f in flags:
                    if f == 'postsynaptic':
                        locality = 'semiglobal'
                    else:
                        locality = 'global'
                    break
            else:
                locality = 'local'

            # Store the result
            desc = {'name': name,
                    'locality': locality,
                    'eq': equation,
                    'bounds': bounds,
                    'flags' : flags,
                    'ctype' : ctype,
                    'init' : init,
                    }
            result.append(desc)
    else:
        Messages._error(f"parameters {description} should either be a dictionary or a multi-string.")

    return result

def extract_variables(description, object_type="neuron", net_id=0):
    """
    Extracts all variable information from a multiline description.

    object_type is either 'neuron' or 'synapse', to get the right locality keywords. (yes, we map global to population/projection back to global...)
    """
    
    
    # If it is a list, convert it to the old multistring. Future versions should reverse the process.
    if isinstance(description, (Variable,)):
        
        description = description._to_string(object_type)
    
    elif isinstance(description, (list,)):

        string_description = ""

        for eq in description:

            # If it is a string, just append it
            if isinstance(eq, (str,)):
                string_description += "\n" + eq
            elif isinstance(eq, (Variable,)):
                string_description += "\n" + eq._to_string(object_type)

        description = string_description
    elif isinstance (description, (str,)):
        pass
    else:
        Messages._error("equations must be either a string or a list of strings/variables.")
    

    # Split the multilines into individual lines
    variable_list = process_equations(description)
    
    # Analyse all variables
    result = []
    for definition in variable_list:

        # Retrieve the name, equation and constraints for the variable
        equation = definition['eq']
        constraint = definition['constraint']
        name = definition['name']
        if name == '_undefined':
            Messages._error('The variable', name, 'can not be analysed.')

        # Check the validity of the equation
        check_equation(equation)

        # Process constraint
        bounds, flags, ctype, init = extract_boundsflags(constraint, net_id=net_id)

        # Determine locality
        for f in ['population', 'postsynaptic', 'projection']:
            if f in flags:
                if f == 'postsynaptic':
                    locality = 'semiglobal'
                else:
                    locality = 'global'
                break
        else:
            locality = 'local'

        # Store the result
        desc = {'name': name,
                'locality': locality,
                'eq': equation,
                'bounds': bounds,
                'flags' : flags,
                'ctype' : ctype,
                'init' : init }
        result.append(desc)

    return result

def extract_boundsflags(constraint, equation="", extra_values={}, net_id=0):
    """
    Implementation note on boolean values:

    The selected C++ data type to represent boolean values differs for CPU and GPU. The reason
    is that a std::vector<bool> is a specialized implementation within the C++ standard template
    library (see https://en.cppreference.com/w/cpp/container/vector_bool).

    According to the reference, this vector "does not necessarily store its elements
    as a contiguous array". Therefore, the access to the raw data using the
    std::vector::data() method is not allowed which would be required to implement
    a host-to-device / device-to-host memory transfer.

    Therefore, we use the following rule: `bool` for single-thread and openMP and `char`
    for CUDA.
    """

    # Process the flags if any
    bounds, flags = extract_flags(constraint)

    # Get the type of the variable (float/int/bool).
    if 'int' in flags:
        ctype = 'int'
    elif 'bool' in flags:
        ctype = 'bool' if _check_paradigm("openmp", net_id) else 'char'
    else:
        ctype = ConfigManager().get('precision', net_id)

    # Detect the random distributions
    random_pattern = r'(' + '|'.join(available_distributions) + r')\s*\([^)]*\)'

    # Get the init value if declared
    if 'init' in bounds.keys(): # Variables: explicitely set in init=xx
        init = bounds['init']
        if ctype == 'bool':
            if init in ['false', 'False', '0']:
                init = False
            elif init in ['true', 'True', '1']:
                init = True
        elif isinstance(init, str) and re.search(random_pattern, init) is not None:
            init = eval(init)
        else:
            try:
                if ctype == 'int':
                    init = int(init)
                else:
                    init = float(init)
            except:
                init = str(init) # Check later whether it is a valid parameter name

    elif '=' in equation: # Parameters: the value is in the equation
        init = equation.split('=')[1].strip()

        # Boolean
        if init in ['false', 'False']:
            init = False
            ctype = 'bool' if _check_paradigm("openmp", net_id) else 'char'
        elif init in ['true', 'True']:
            init = True
            ctype = 'bool' if _check_paradigm("openmp", net_id) else 'char'
        # Extra-args (obsolete)
        elif init.strip().startswith("'"):
            var = init.replace("'","")
            init = extra_values[var]
        # Integers
        elif ctype == 'int':
            try:
                init = eval('int(' + init + ')')
            except:
                Messages._print(equation)
                Messages._error('The value of the parameter is not an integer.')
        # Floats
        else:
            try:
                init = eval('float(' + init + ')')
            except:
                Messages._print(equation)
                Messages._error('The value of the parameter is not a float.')

    else: # Default = 0 according to ctype
        if ctype == 'bool':
            init = False
        elif ctype == 'int':
            init = 0
        elif ctype == 'double' or ctype == 'float':
            init = 0.0

    return bounds, flags, ctype, init

def extract_functions(description, net_id, local_global=False):
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

        # Check the function name is not reserved by Sympy
        from inspect import getmembers
        import sympy
        functions_list = [o[0] for o in getmembers(sympy)]
        if func_name in functions_list:
            Messages._error('The function name', func_name, 'is reserved by sympy. Use another one.')

        # Extract their types
        types = f['constraint']
        if types == '':
            return_type = ConfigManager().get('precision', net_id)
            arg_types = [ConfigManager().get('precision', net_id) for a in arguments]
        else:
            types = types.split(',')
            return_type = types[0].strip()
            arg_types = [arg.strip() for arg in types[1:]]
        if not len(arg_types) == len(arguments):
            Messages._error('You must specify exactly the types of return value and arguments in ' + eq)

        arg_line = ""
        for i in range(len(arguments)):
            arg_line += arg_types[i] + " " + arguments[i]
            if not i == len(arguments) -1:
                arg_line += ', '

        # Process the content
        eq2, condition = extract_ite('', content, {'attributes': [], 'local':[], 'global': [], 'variables': [], 'parameters': []}, split=False)
        if condition == []:
            parser = FunctionParser('', content, arguments)
            parsed_content = parser.parse()
        else:
            parsed_content, deps = translate_ITE("", eq2, condition, arguments, {}, function=True)
            arguments = list(set(arguments)) # somehow the entries in arguments are doubled ... ( HD, 23.02.2017 )

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


def get_attributes(parameters, variables, neuron):
    """ 
    Returns a list of all attributes names, plus the lists of local/global variables.
    """
    attributes = []; local_var = []; global_var = []; semiglobal_var = []

    for p in parameters + variables:
        attributes.append(p['name'])
        if neuron:
            if 'population' in p['flags']:
                global_var.append(p['name'])
            elif 'projection' in p['flags']:
                Messages._error('The attribute', p['name'], 'belongs to a neuron, the flag "projection" is forbidden.')
            elif 'postsynaptic' in p['flags']:
                Messages._error('The attribute', p['name'], 'belongs to a neuron, the flag "postsynaptic" is forbidden.')
            else:
                local_var.append(p['name'])
        else:
            if 'population' in p['flags']:
                Messages._error('The attribute', p['name'], 'belongs to a synapse, the flag "population" is forbidden.')
            elif 'projection' in p['flags']:
                global_var.append(p['name'])
            elif 'postsynaptic' in p['flags']:
                semiglobal_var.append(p['name'])
            else:
                local_var.append(p['name'])

    return attributes, local_var, global_var, semiglobal_var

def extract_targets(variables):
    targets = []
    for var in variables:
        # Rate-coded neurons
        code = re.findall(r'(?P<pre>[^\w.])sum\(\s*([^()]+)\s*\)', var['eq'])
        for l, t in code:
            targets.append(t.strip())
        # Special case for sum()
        if len(re.findall(r'([^\w.])sum\(\)', var['eq'])) > 0:
            targets.append('__all__')

        # Spiking neurons
        code = re.findall(r'([^\w.])g_([\w]+)', var['eq'])
        for l, t in code:
            targets.append(t.strip())

    return list(set(targets))


def convert_to_multistring(equations):
    """
    Converts a list of variables into a multistring that can be passed to process_equations()
    """
        
    # If it is a list, convert it to the old multistring. Future versions should reverse the process.
    if isinstance(equations, (Variable,)):
        
        equations = equations._to_string('neuron')
    
    elif isinstance(equations, (list,)):

        string_equations = ""

        for eq in equations:

            # If it is a string, just append it
            if isinstance(eq, (str,)):
                string_equations += "\n" + eq
            elif isinstance(eq, (Variable,)):
                string_equations += "\n" + eq._to_string('neuron')
        
        equations = string_equations

    elif isinstance (equations, (str,)):
        # Already a string, all good
        pass

    else:
        Messages._error("equations must be either a string or a list of strings/variables.")

    return equations

def extract_spike_variable(description, net_id):

    # Spike condition
    cond = prepare_string(description['raw_spike'])
    if len(cond) > 1:
        Messages._print(description['raw_spike'])
        Messages._error('The spike condition must be a single expression')

    translator = Equation('raw_spike_cond',
                            cond[0].strip(),
                            description)
    raw_spike_code = translator.parse()
    # Also store the variables used in the condition, as it may be needed for CUDA generation
    spike_code_dependencies = translator.dependencies()

    # Reset
    reset_desc = []
    if 'raw_reset' in description.keys() and description['raw_reset']:

        # Get equations
        equations = description['raw_reset']

        # Convert the equations to a multistring
        equations = convert_to_multistring(equations)
        
        # Process each line
        reset_desc = process_equations(equations)
        
        for var in reset_desc:
            translator = Equation(var['name'], var['eq'],
                                  description)
            var['cpp'] = translator.parse()
            var['dependencies'] = translator.dependencies()

    return { 'spike_cond': raw_spike_code,
             'spike_cond_dependencies': spike_code_dependencies,
             'spike_reset': reset_desc}

def extract_axon_spike_condition(description, net_id):
    """
    Extract the condition for emitting an axonal spike event. Further
    the reset after the event is returned.
    """
    if description['raw_axon_spike'] == None:
        return None

    cond = prepare_string(description['raw_axon_spike'])
    if len(cond) > 1:
        Messages._print(description['raw_axon_spike'])
        Messages._error('The spike condition must be a single expression')

    translator = Equation('raw_axon_spike_cond',
                            cond[0].strip(),
                            description)
    raw_spike_code = translator.parse()
    # Also store the variables used in the condition, as it may be needed for CUDA generation
    spike_code_dependencies = translator.dependencies()

    reset_desc = []
    if 'raw_reset' in description.keys() and description['raw_axon_reset']:

        # Convert the equations to a multistring
        equations = convert_to_multistring(equations)

        # Process the equations
        reset_desc = process_equations(description['raw_axon_reset'])
        for var in reset_desc:
            translator = Equation(var['name'], var['eq'],
                                  description)
            var['cpp'] = translator.parse()
            var['dependencies'] = translator.dependencies()

    return {
        'spike_cond': raw_spike_code,
        'spike_cond_dependencies': spike_code_dependencies,
        'spike_reset': reset_desc
    }

def extract_pre_spike_variable(description, net_id):

    # Get the equations
    equations = description['raw_pre_spike']
    if equations is None:
        return []

    # Convert the equations to a multistring
    equations = convert_to_multistring(equations)

    # For all variables influenced by a presynaptic spike
    pre_spike_var = []
    for var in process_equations(equations):
        # Get its name
        name = var['name']
        eq = var['eq']

        # Process the flags if any
        bounds, flags, ctype, init = extract_boundsflags(var['constraint'], net_id=net_id)

        # Extract if-then-else statements
        #eq, condition = extract_ite(name, raw_eq, description)

        # Append the result of analysis
        pre_spike_var.append( { 'name': name, 'eq': eq ,
                                'locality': 'local',
                                'bounds': bounds,
                                'flags':flags, 'ctype' : ctype,
                                'init' : init} )

    return pre_spike_var

def extract_post_spike_variable(description, net_id):
    
    # Get the equations
    equations = description['raw_post_spike']
    if not equations:
        return []

    # Convert the equations to a multistring
    equations = convert_to_multistring(equations)

    post_spike_var = []
    for var in process_equations(equations):
        # Get its name
        name = var['name']
        eq = var['eq']

        # Process the flags if any
        bounds, flags, ctype, init = extract_boundsflags(var['constraint'], net_id=net_id)

        # Extract if-then-else statements
        #eq, condition = extract_ite(name, raw_eq, description)

        post_spike_var.append( { 'name': name, 'eq': eq, 'raw_eq' : eq,
                                'locality': 'local',
                                'bounds': bounds, 'flags':flags, 'ctype' : ctype, 'init' : init} )

    return post_spike_var

def extract_axon_spike_variable(description, net_id):
    
    # Get the equations
    equations = description['raw_axon_spike']
    if not equations: 
        return []

    # Convert the equations to a multistring
    equations = convert_to_multistring(equations)

    # For all variables influenced by a presynaptic spike
    axon_spike_var = []
    for var in process_equations(equations):
        # Get its name
        name = var['name']
        eq = var['eq']

        # Process the flags if any
        bounds, flags, ctype, init = extract_boundsflags(var['constraint'], net_id=net_id)

        # Append the result of analysis
        axon_spike_var.append( {'name': name, 'eq': eq ,
                                'locality': 'local',
                                'bounds': bounds,
                                'flags':flags, 'ctype' : ctype,
                                'init' : init} )

    return axon_spike_var

def extract_stop_condition(pop, net_id=0):

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
                          pop,
                          type = 'cond')
    code = translator.parse()
    deps = translator.dependencies()

    pop['stop_condition']['cpp'] = '(' + code + ')'
    pop['stop_condition']['dependencies'] = deps

def extract_structural_plasticity(statement, description, net_id=0):

    # Extract flags
    if isinstance(statement, (str,)):
        try:
            eq, constraint = statement.rsplit(':', 1)
            bounds, flags = extract_flags(constraint)
        except:
            eq = statement.strip()
            bounds = {}
            flags = []

    elif isinstance(statement, (Creating)):
        eq = statement.equation
        proba = statement.proba
        w = statement.w
        d = statement.d
        bounds = {'proba': str(proba), 'w': str(w)}
        if d is not None: bounds['d'] = str(d)
        flags = []

    elif isinstance(statement, (Pruning)):
        eq = statement.equation
        proba = statement.proba
        bounds = {'proba': str(proba)}
        flags = []

    # Extract RD
    rd = None
    for dist in available_distributions:
        matches = re.findall(r'(?P<pre>[^\w.])' + dist + r'\(([^()]+)\)', eq)
        for l, v in matches:
            # Check the arguments
            arguments = v.split(',')
            # Check the number of provided arguments
            if len(arguments) < distributions_arguments[dist]:
                Messages._print(eq)
                Messages._error('The distribution ' + dist + ' requires ' + str(distributions_arguments[dist]) + 'parameters')
            elif len(arguments) > distributions_arguments[dist]:
                Messages._print(eq)
                Messages._error('Too many parameters provided to the distribution ' + dist)
            # Process the arguments
            processed_arguments = ""
            for idx in range(len(arguments)):
                try:
                    arg = float(arguments[idx])
                except: # A global parameter
                    Messages._print(eq)
                    Messages._error('Random distributions for creating/pruning synapses must use foxed values.')

                processed_arguments += str(arg)
                if idx != len(arguments)-1: # not the last one
                    processed_arguments += ', '
            definition = distributions_equivalents[dist] + '(' + processed_arguments + ')'

            # Store its definition
            if rd:
                Messages._print(eq)
                Messages._error('Only one random distribution per equation is allowed.')


            rd = {
                'name': 'rand_' + str(0) ,
                'origin': dist+'('+v+')',
                'dist': dist,
                'definition': definition,
                'args' : processed_arguments,
                'template': distributions_equivalents[dist]
            }

    if rd:
        eq = eq.replace(rd['origin'], 'rd(rng)')

    # Extract pre/post dependencies
    eq, untouched, dependencies = extract_prepost('test', eq, description, net_id=net_id)

    # Parse code
    translator = Equation('test', eq,
                          description,
                          method = 'cond',
                          untouched = {})

    code = translator.parse()
    deps = translator.dependencies()

    # Replace untouched variables with their original name
    for prev, new in sorted(list(untouched.items()), key = lambda key : len(key[0]), reverse=True):
        code = code.replace(prev, new)

    # Add new dependencies
    for dep in dependencies['pre']:
        description['dependencies']['pre'].append(dep)
    for dep in dependencies['post']:
        description['dependencies']['post'].append(dep)

    return {'eq': eq, 'cpp': code, 'bounds': bounds, 'flags': flags, 'rd': rd, 'dependencies': deps}


def find_method(variable):

    if 'implicit' in variable['flags']:
        method = 'implicit'
    elif 'semiimplicit' in variable['flags']:
        method = 'semiimplicit'
    elif 'exponential' in variable['flags']:
        method = 'exponential'
    elif 'midpoint' in variable['flags']:
        method = 'midpoint'
    elif 'runge-kutta4' in variable['flags']: # old name, backward compatibility
        method = 'rk4'
    elif 'rk4' in variable['flags']:
        method = 'rk4'
    elif 'explicit' in variable['flags']:
        method = 'explicit'
    elif 'exact' in variable['flags']:
        Messages._warning('The "exact" flag should now be replaced by "event-driven". It will stop being valid in a future release.')
        method = 'event-driven'
    elif 'event-driven' in variable['flags']:
        method = 'event-driven'
    else:
        method = get_global_config('method')
        if method == "runge-kutta4": # old name, backward compatibility
            method = 'rk4'

    return method

def check_equation(equation):
    "Makes a formal check on the equation (matching parentheses, etc)"
    # Matching parentheses
    if equation.count('(') != equation.count(')'):
        Messages._print(equation)
        Messages._error('The number of parentheses does not match.')
