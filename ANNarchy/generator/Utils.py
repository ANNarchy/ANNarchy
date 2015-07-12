from ..core.Global import config
 
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
                    if len(equations) ==0:
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
        'index': '%(local_index)s' if param['locality'] == 'local' else '%(global_index)s',
        'var' : param['name'], 'val' : val, 
        'operator': '<' if bound=='min' else '>'
    }
    return code

def generate_non_ODE_block(variables, locality, obj, conductance_only):
    code = ""
    for param in variables: 
        if conductance_only: # skip the variables which do not start with g_
            if not param['name'].startswith('g_'):
                continue
        code += """
%(comment)s
%(cpp)s
%(bounds)s
""" % { 'comment': '// ' + param['eq'],
        'cpp': param['cpp'],
        'bounds': generate_bound_code(param, obj) }

    return code


def generate_ODE_block(odes, locality, obj, conductance_only):
    code = ""

    # Count how many steps (midpoint has more than one step)
    nb_step = 0
    for param in odes: 
        if isinstance(param['cpp'], list):
            nb_step = max(len(param['cpp']), nb_step)
        else:
            nb_step = max(1, nb_step)

    # Iterate over all steps
    if len(odes) > 0:
        for step in range(nb_step):
            for param in odes:
                if conductance_only: # skip the variables which do not start with g_
                    if not param['name'].startswith('g_'):
                        continue
                if isinstance(param['cpp'], list) and step < len(param['cpp']):
                    eq = param['cpp'][step]
                elif isinstance(param['cpp'], str) and step == 0: 
                    eq = param['cpp']
                else:
                    eq = ''
                code += """
%(comment)s
%(cpp)s
""" % { 'comment': '// '+param['eq'],
                'cpp': eq }

        # Generate the switch code
        for param in odes: 
            if conductance_only: # skip the variables which do not start with g_
                if not param['name'].startswith('g_'):
                    continue
            code += """
%(comment)s
%(switch)s 
""" % { 'comment': '// '+param['eq'],
        'switch' : param['switch']}

            # Min-Max bounds,
            code += generate_bound_code(param, obj)

    return code

def generate_equation_code(id, desc, locality='local', obj='pop', conductance_only=False, padding=3):
    
    # Separate ODEs from the pre- and post- equations
    odes = sort_odes(desc, locality)
    
    if odes == []: # No equations
        return ""

    # Generate code
    code = ""  

    for type_block, block in odes:
        if type_block == 'ode':
            code += generate_ODE_block(block, locality, obj, 
            conductance_only)
        else:
            code += generate_non_ODE_block(block, locality, obj, 
            conductance_only)

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
