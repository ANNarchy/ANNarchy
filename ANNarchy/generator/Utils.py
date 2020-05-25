#===============================================================================
#
#     Utils.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core import Global
import re

def sort_odes(desc, locality='local'):
    equations = []
    is_ode = False
    for param in desc['variables']:
        if param['cpp'] == '':
            continue
        if param['method'] == 'event-driven':
            continue
        if param['name'] in desc[locality]:
            if param['switch']: # ODE
                if is_ode: # was already ODE
                    if len(equations) == 0:
                        equations.append(('ode', [param]))
                    else:
                        equations[-1][1].append(param)
                else: # new block
                    is_ode = True
                    equations.append(('ode', [param]))
            else: # non-ODE
                if is_ode:
                    is_ode = False
                    equations.append(('non-ode', [param]))
                else:
                    if len(equations) == 0:
                        equations.append(('non-ode', [param]))
                    else:
                        equations[-1][1].append(param)

    return equations

def generate_bound_code(param, obj):
    code = ""
    for bound, val in param['bounds'].items():
        if bound in ['min', 'max']:
            code += """if(%(var)s%(index)s %(operator)s %(val)s)
    %(var)s%(index)s = %(val)s;
""" % {
        'index': '%(local_index)s' if param['locality'] == 'local' else ('%(semiglobal_index)s' if param['locality'] == 'semiglobal' else'%(global_index)s'),
        'var' : param['name'], 'val' : val,
        'operator': '<' if bound=='min' else '>'
    }
    return code

def generate_non_ODE_block(variables, locality, obj, conductance_only, wrap_w, split_loop=False):
    " TODO: documentation "
    block_code = ""
    block_bounds = ""
    for param in variables:
        if conductance_only: # skip the variables which do not start with g_
            if not param['name'].startswith('g_'):
                continue
        bounds = generate_bound_code(param, obj)
        if wrap_w and param['name'] == "w":
            block_code += """
%(comment)s
if(%(wrap)s){
%(cpp)s
%(bounds)s
}
""" % { 'comment': '// ' + param['eq'],
        'cpp': param['cpp'],
        'wrap': wrap_w,
        'bounds': bounds if not split_loop else ""}
            block_bounds += bounds if split_loop else ""

        else:
            block_code += """
%(comment)s
%(cpp)s
%(bounds)s
""" % { 'comment': '// ' + param['eq'],
        'cpp': param['cpp'],
        'bounds': bounds if not split_loop else "" }
            block_bounds += bounds if split_loop else ""

    if not split_loop:
        return block_code
    else:
        return block_code, block_bounds


def generate_ODE_block(odes, locality, obj, conductance_only, wrap_w):
    code = ""

    # Count how many steps (midpoint has more than one step)
    nb_step = 0
    for param in odes:
        if isinstance(param['cpp'], list):
            nb_step = max(len(param['cpp']), nb_step)
        else:
            nb_step = max(1, nb_step)

    if len(odes) == 0:
        return ""

    # Iterate over all steps
    for step in range(nb_step):
        for param in odes:
            if conductance_only: # skip the variables which do not start with g_
                if not param['name'].startswith('g_'):
                    continue
            # Retrieve equation
            if isinstance(param['cpp'], list) and step < len(param['cpp']):
                eq = param['cpp'][step]
            elif isinstance(param['cpp'], str) and step == 0:
                eq = param['cpp']
            else:
                eq = ''
            # Generate code
            code += """
%(comment)s
%(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': eq }

    # Generate the switch code
    for param in odes:
        bounds = generate_bound_code(param, obj)
        if conductance_only: # skip the variables which do not start with g_
            if not param['name'].startswith('g_'):
                continue
        if wrap_w and param['name'] == "w":
            code += """
%(comment)s
if(%(wrap)s){
%(switch)s
%(bounds)s
}
""" % { 'comment': '// '+param['eq'],
        'wrap': wrap_w,
        'bounds': bounds,
        'switch' : param['switch']}
        else:
            code += """
%(comment)s
%(switch)s
%(bounds)s
""" % { 'comment': '// '+param['eq'],
        'bounds': bounds,
        'switch' : param['switch']}


    return code

def generate_equation_code(obj_id, desc, locality='local', obj='pop', conductance_only=False, wrap_w=None, padding=3):
    """ TODO: 
    * documentation
    * do we really need the obj_id (former pop_id) ?
    """
    # Separate ODEs from the pre- and post- equations
    odes = sort_odes(desc, locality)

    if odes == []: # No equations
        return ""

    # Generate code
    code = ""
    for type_block, block in odes:
        if type_block == 'ode':
            code += generate_ODE_block(block, locality, obj, conductance_only, wrap_w)
        elif type_block == 'non-ode':
            code += generate_non_ODE_block(block, locality, obj, conductance_only, wrap_w, split_loop=False)
        else:
            raise NotImplementedError

    # Add the padding to each line
    padded_code = tabify(code, padding)

    return padded_code

def indentLine(line, spaces=1):
    return (' ' * 4 * spaces) + line

def tabify(s, numSpaces):
    s = s.split('\n')
    s = map(lambda a, ns=numSpaces: indentLine(a, ns), s)
    s = '\n'.join(s)
    return s

def remove_trailing_spaces(code):
    """
    The generated code templates often contain empty lines, which are indented by tabify() or indentLine()
    afterwards which this introduces many white spaces which are annoying in some editors. The call of rstrip()
    on the complete string can not remove them. Therefore we implement this little helper function to call
    rstrip on each line.
    """
    stripped_lines = [line.rstrip() for line in code.split('\n')]

    stripped_code = ""
    for line in stripped_lines:
        stripped_code += line +'\n'

    return stripped_code

def check_and_apply_pow_fix(eqs):
    """
    CUDA SDKs before 7.5 had an error if std=c++11 is enabled related
    to pow(double, int). Only pow(double, double) was detected as
    device function, the pow(double, int) will be detected as host
    function. (This was fixed within SDK 7.5)

    To support also earlier versions, we simply add a double type cast.
    """
    if eqs.strip() == "":
        # nothing to do
        return eqs

    try:
        from ANNarchy.generator.CudaCheck import CudaCheck
        if CudaCheck().runtime_version() > 7000:
            # nothing to do, is working in higher SDKs
            return eqs
    except:
        Global._error('CUDA is not correctly installed on your system')

    if Global.config['verbose']:
        Global._print('occurance of pow() and SDK below 7.5 detected, apply fix.')

    # detect all pow statements
    pow_occur = re.findall(r"pow[\( [\S\s]*?\)*?, \d+\)]*?", eqs)
    for term in pow_occur:
        eqs = eqs.replace(term, term.replace(', ', ', (double)'))

    return eqs
