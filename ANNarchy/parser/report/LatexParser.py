from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number
import re

import ANNarchy.core.Global as Global
from ANNarchy.core.Random import RandomDistribution
from ..Extraction import *
from ANNarchy.parser.AnalyseSynapse import analyse_synapse


##################################
### Process individual equations
##################################

def _process_random(val):
    "Transforms a connector attribute (weights, delays) into a string representation"
    if isinstance(val, RandomDistribution):
        return val.latex()
    else:
        return str(val)

# Really crappy...
# When target has a number (ff1), sympy thinks the 1 is a number
# the target is replaced by a text to avoid this
target_replacements = [
    'firsttarget',
    'secondtarget',
    'thirdtarget',
    'fourthtarget',
    'fifthtarget',
    'sixthtarget',
    'seventhtarget',
    'eighthtarget',
    'ninthtarget',
    'tenthtarget',
]

def _process_neuron_equations(neuron):
    code = ""

    # Extract parameters and variables
    parameters = extract_parameters(neuron.parameters, neuron.extra_values)
    variables = extract_variables(neuron.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, semiglobal_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'g_target': Symbol('g_{\\text{target}}'),
        't_pre': Symbol('t_{\\text{pre}}'),
        't_post': Symbol('t_{\\text{pos}}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.items():
        tex_dict[val] = str(val)

    for var in variables:
        # Retrieve the equation
        eq = var['eq']
        eq = eq.replace(' ', '') # supress spaces

        # Extract sum(target)
        targets = []
        target_list = re.findall('(?P<pre>[^\w.])sum\(\s*([^()]+)\s*\)', eq)
        for l, t in target_list:
            if t.strip() == '':
                continue
            targets.append((t.strip(), target_replacements[len(targets)]))
        for target, repl in targets:
            eq = eq.replace('sum('+target+')', repl)

        # Parse the equation
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)

        # Replace the targets
        for target, repl in targets:
            var_code = var_code.replace(repl, '\\sum_{\\text{'+target+'}} \\text{psp}(t)')

        # Add the code
        var['latex'] = var_code
        var['ode'] = len(ode) > 0

    if not neuron.spike: # rate-code, no spike
        return variables, "", []

    # Additional code for spiking neurons
    spike_condition = _analyse_part(neuron.spike, local_dict, tex_dict)

    # Reset
    spike_reset = []
    reset_vars = extract_variables(neuron.reset)
    for var in reset_vars:
        eq = var['eq']
        spike_reset.append(_analyse_equation(var['eq'], eq, local_dict, tex_dict))

    return variables, spike_condition, spike_reset


def _process_synapse_equations(synapse):
    psp = ""
    code = ""
    pre_event = []
    post_event = []

    # Extract parameters and variables
    parameters = extract_parameters(synapse.parameters)
    variables = extract_variables(synapse.equations)
    variable_names = [var['name'] for var in variables]
    attributes, local_var, semiglobal_var, global_var = get_attributes(parameters, variables)

    # Create a dictionary for parsing
    local_dict = {
        'w': Symbol('w(t)'),
        'g_target': Symbol('g_{\\text{target}(t)}'),
        't_pre': Symbol('t_{\\text{pre}}'),
        't_post': Symbol('t_{\\text{pos}}'),
        'Uniform': Function('\mathcal{U}'),
        'Normal': Function('\mathcal{N}'),
    }

    for att in attributes:
        local_dict[att] = Symbol(_latexify_name(att, variable_names))

    tex_dict = {}
    for key, val in local_dict.items():
        tex_dict[val] = str(val)


    # PSP
    if synapse.psp:
        psp, untouched_var, dependencies = extract_prepost('psp', synapse.psp.strip(), synapse.description)
        for dep in dependencies['post']:
            local_dict['_post_'+dep+'__'] = Symbol("{" + dep + "^{\\text{post}}}(t)")
        for dep in dependencies['pre']:
            local_dict['_pre_'+dep+'__'] = Symbol("{" + dep + "^{\\text{pre}}}(t)")
        if synapse.type == 'rate':  
            psp = "$$\\text{sum(target)}(t) \mathrel{+}= " + _analyse_part(psp, local_dict, tex_dict) + "$$"
        else:  
            psp = "$$g_\\text{target}(t) \mathrel{+}= " + _analyse_part(psp, local_dict, tex_dict) + "$$"
    else:
        if synapse.type == 'rate':
            psp = "$$\\text{sum(target)}(t) \mathrel{+}= w(t) \cdot r^{\\text{pre}}(t)$$"
        else:
            psp = ""


    # Variables
    for var in variables:
        # Retrieve the equation
        eq = var['eq']

        # pre/post variables
        targets=[]
        eq, untouched_var, dependencies = extract_prepost(var['name'], eq, synapse.description)

        for dep in dependencies['post']:
            if dep.startswith('sum('):
                target = re.findall(r'sum\(([\w]+)\)', dep)[0]
                targets.append(target)
                local_dict['_post_sum_'+target] = Symbol('PostSum'+target)
            else:
                local_dict['_post_'+dep+'__'] = Symbol("{" + dep + "^{\\text{post}}}(t)")

        for dep in dependencies['pre']:
            if dep.startswith('sum('):
                target = re.findall(r'sum\(([\w]+)\)', dep)[0]
                targets.append(target)
                local_dict['_pre_sum_'+target] = Symbol('PreSum'+target)
            else:
                local_dict['_pre_'+dep+'__'] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

        # Parse the equation
        eq = eq.replace(' ', '') # supress spaces
        ode = re.findall(r'([^\w]*)d([\w]+)/dt', eq)
        if len(ode) > 0:
            name = ode[0][1]
            eq = eq.replace('d'+name+'/dt', '_grad_'+name)
            grad_symbol = Symbol('\\frac{d'+_latexify_name(name, variable_names)+'}{dt}')
            local_dict['_grad_'+name] = grad_symbol
            tex_dict[grad_symbol] = '\\frac{d'+_latexify_name(name, variable_names)+'}{dt}'

        # Analyse
        var_code = _analyse_equation(var['eq'], eq, local_dict, tex_dict)

        # replace targets
        for target in targets:
            var_code = var_code.replace('PostSum'+target, "(\\sum_{\\text{" + target + "}} \\text{psp}(t))^{\\text{post}}")
            var_code = var_code.replace('PreSum'+target,  "(\\sum_{\\text{" + target + "}} \\text{psp}(t))^{\\text{pre}}")

        # Add the code
        var['latex'] = var_code
        var['ode'] = len(ode) > 0

    # Pre-event
    if synapse.type == 'spike':
        desc = analyse_synapse(synapse)
        for var in extract_pre_spike_variable(desc):
            eq = var['eq']
            # pre/post variables
            eq, untouched_var, dependencies = extract_prepost(var['name'], eq, desc)
            for dep in dependencies['post']:
                local_dict['_post_'+dep+'__'] = Symbol("{" + dep + "^{\\text{post}}}(t)")
            for dep in dependencies['pre']:
                local_dict['_pre_'+dep+'__'] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

            pre_event.append(_analyse_equation(var['eq'], eq, local_dict, tex_dict))

        for var in extract_post_spike_variable(desc):
            eq = var['eq']
            # pre/post variables
            eq, untouched_var, dependencies = extract_prepost(var['name'], eq, desc)
            for dep in dependencies['post']:
                local_dict['_post_'+dep+'__'] = Symbol("{" + dep + "^{\\text{post}}}(t)")
            for dep in dependencies['pre']:
                local_dict['_pre_'+dep+'__'] = Symbol("{" + dep + "^{\\text{pre}}}(t)")

            post_event.append(_analyse_equation(var['eq'], eq, local_dict, tex_dict))

    return psp, variables, pre_event, post_event


def _process_functions(functions, begin="\\begin{dmath*}\n", end="\n\\end{dmath*}"):
    code = ""

    extracted_functions = extract_functions(functions, False)

    for func in extracted_functions:
        # arguments
        args = func['args']
        args_list = ""
        for arg in args:
            args_list += _latexify_name(arg, []) + ", "
        args_list = args_list[:-2]

        # local dict
        local_dict = {}
        for att in args:
            local_dict[att] = Symbol(_latexify_name(att, []))
        tex_dict = {}
        for key, val in local_dict.items():
            tex_dict[val] = str(val)

        # parse the content
        content = _analyse_part(func['content'], local_dict, tex_dict)

        # generate the code
        code += "%(begin)s%(name)s(%(args)s) = %(content)s%(end)s" %  {
            'name': _latexify_name(func['name'], []), 
            'args': args_list, 
            'content': content.strip(), 
            'begin': begin, 
            'end': end }


    return code


# Splits an equation into two parts, caring for the increments
def _analyse_equation(orig, eq, local_dict, tex_dict):

    # Analyse the left part
    left = eq.split('=')[0]
    split_idx = len(left)
    if left[-1] in ['+', '-', '*', '/']:
        op = left[-1]
        try:
            left = _analyse_part(left[:-1], local_dict, tex_dict)
        except Exception as e:
            Global._print(e)
            Global._warning('can not transform the left side of ' + orig +' to LaTeX, you have to do it by hand...')
            left = left[:-1]
        operator = " = " + left +  " " + op + (" (" if op != '+' else '')
        operator = " \mathrel{" + op + "}= " 
    else:
        try:
            left = _analyse_part(left, local_dict, tex_dict)
        except Exception as e:
            Global._print(e)
            Global._warning('can not transform the left side of ' + orig +' to LaTeX, you have to do it by hand...')
        operator = " = "

    # Analyse the right part
    try:
        right = _analyse_part(eq[split_idx+1:], local_dict, tex_dict)
    except Exception as e:
        Global._print(e)
        Global._warning('can not transform the right side of ' + orig +' to LaTeX, you have to do it by hand...')
        right = "%%" + eq[split_idx+1:]

    return left + operator + right + (" )" if operator.endswith('(') else "")


# Analyses and transform to latex a single part of an equation
def _analyse_part(expr, local_dict, tex_dict):

    def regular_expr(expr):
        analysed = parse_expr(expr,
            local_dict = local_dict,
            transformations = (standard_transformations + (convert_xor,))
            )
        return latex(analysed, symbol_names = tex_dict, mul_symbol="dot")

    def _condition(condition):
        condition = condition.replace('and', ' & ')
        condition = condition.replace('or', ' | ')
        return regular_expr(condition)

    def _extract_conditional(condition):
        if_statement = condition[0]
        then_statement = condition[1]
        else_statement = condition[2]
        # IF condition
        if_code = _condition(if_statement)
        # THEN
        if isinstance(then_statement, list): # nested conditional
            then_code =  _extract_conditional(then_statement)
        else:
            then_code = regular_expr(then_statement)
        # ELSE
        if isinstance(else_statement, list): # nested conditional
            else_code =  _extract_conditional(else_statement)
        else:
            else_code = regular_expr(else_statement)
        return "\\begin{cases}" + then_code + "\qquad \\text{if} \quad " + if_code + "\\\\ "+ else_code +" \qquad \\text{otherwise.} \end{cases}"

    # Extract if/then/else
    if 'else:' in expr:
        ite_code = extract_ite(expr)
        return _extract_conditional(ite_code)

    # return the transformed equation
    return regular_expr(expr)

def extract_ite(eq):
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

    # Process the equation
    multilined = transform(eq)
    condition = parse(multilined)
    return condition

# Latexify names
greek = ['alpha', 'beta', 'gamma', 'epsilon', 'eta', 'kappa', 'delta', 'lambda', 'mu', 'nu', 'zeta', 'sigma', 'phi', 'psi', 'rho', 'omega', 'xi', 'tau',
         'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Phi', 'Psi', 'Omega'
]

def _latexify_name(name, local):
    parts = name.split('_')
    if len(parts) == 1:
        if len(name) == 1:
            equiv = name
        elif name in greek:
            equiv = '\\' + name
        else:
            equiv = '{\\text{' + name + '}}'
        if name in local:
            equiv = '{' + equiv + '}(t)'
        return equiv
    elif len(parts) == 2:
        equiv = ""
        for p in parts:
            if len(p) == 1:
                equiv += '' + p + '_'
            elif p in greek:
                equiv += '\\' + p + '_'
            else:
                equiv += '{\\text{' + p + '}}' + '_'
        equiv = equiv[:-1]
        if name in local:
            equiv = '{' + equiv + '}(t)'
        return equiv
    else:
        equiv = '{\\text{' + name + '}}'
        equiv = equiv.replace('_', '\_')
        if name in local:
            equiv = equiv + '(t)'
        return equiv

def pop_name(name):
    return name.replace('_', '\_')


def _format_list(l, sep):
    if not isinstance(l, list):
        return l
    target_list = ""
    for t in l:
        target_list += t + sep
    return target_list[:-len(sep)]
