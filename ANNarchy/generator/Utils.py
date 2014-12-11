from ..parser.SingleAnalysis import pattern_omp, pattern_cuda
from ..core.Global import config
 
def sort_odes(desc, locality='local'):
    pre_odes = []
    odes = []
    post_odes=[]
    is_pre_ode = True
    is_ode = False
    for param in desc['variables']: 
        if param['cpp'] == '':
            continue
        if param['method'] == 'exact':
            continue
        if param['name'] in desc[locality]: 
            if is_pre_ode: # look if it is an ode or not
                if param['switch']: # ODE
                    is_pre_ode = False
                    is_ode = True
                    odes.append(param)
                else: # still pre_ode
                    pre_odes.append(param)
            elif is_ode: # in the ODE section
                if param['switch']: # ODE
                    odes.append(param)
                else: # switch to post_ode
                    is_ode = False
                    post_odes.append(param)
            else: # post_ode
                if param['switch']: # ODE after a post equation, shift it above
                    odes.append(param)
                else:
                    post_odes.append(param)

    return pre_odes, odes, post_odes

def generate_equation_code(id, desc, locality='local', obj='pop'):
    
    # Find the paradigm OMP or CUDA
    if config['paradigm'] == 'openmp':
        pattern = pattern_omp
    else:
        pattern = pattern_cuda
    
    # Separate ODEs from the pre- and post- equations
    pre_odes, odes, post_odes = sort_odes(desc, locality)
    
    if (pre_odes, odes, post_odes) == ([], [], []): # No equations
        return ""

    # Generate code
    code = ""  

    #########################       
    # Pre-ODE equations
    #########################
    if len(pre_odes) > 0:
        code += """
        /////////////////////////
        // Before the ODES
        /////////////////////////
"""
        for param in pre_odes: 
            code += """
        %(comment)s
        %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }
            # Min-Max bounds
            for bound, val in param['bounds'].iteritems():
                code += """
        if(%(obj)s%(sep)s%(var)s%(index)s %(operator)s %(val)s)
            %(obj)s%(sep)s%(var)s%(index)s = %(val)s;
""" % {'obj': pattern['pop_prefix'] if obj == 'pop' else pattern['proj_prefix'],
       'sep': pattern['pop_sep'] if obj == 'pop' else pattern['proj_sep'],
       'index': pattern['pop_index'] if obj == 'pop' else pattern['proj_index'],
       'var' : param['name'], 'val' : val, 'id': id, 
       'operator': '<' if bound=='min' else '>'
       }

    #################
    # ODE equations
    #################
    # Count how many steps (midpoint has more than one step)
    nb_step = 0
    for param in odes: 
        if isinstance(param['cpp'], list):
            nb_step = max(len(param['cpp']), nb_step)
        else:
            nb_step = max(1, nb_step)

    # Iterate over all steps
    if len(odes) > 0:
        code += """
        /////////////////////////
        // ODES
        /////////////////////////
"""
        for step in range(nb_step):
            code += """
        // Step %(step)s
        """ % {'step' : str(step+1)}
            for param in odes:
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
        code += """    
        /////////////////////
        // Switch values
        /////////////////////
"""
        for param in odes: 
            code += """
        %(comment)s
        %(switch)s 
""" % { 'comment': '// '+param['eq'],
        'switch' : param['switch']}
    
            # Min-Max bounds
            for bound, val in param['bounds'].iteritems():
                code += """
        if(%(obj)s%(sep)s%(var)s%(index)s %(operator)s %(val)s)
            %(obj)s%(sep)s%(var)s%(index)s = %(val)s;
""" % {'obj': pattern['pop_prefix'] if obj == 'pop' else pattern['proj_prefix'],
       'sep': pattern['pop_sep'] if obj == 'pop' else pattern['proj_sep'],
       'index': pattern['pop_index'] if obj == 'pop' else pattern['proj_index'],
       'var' : param['name'], 'val' : val, 'id': id, 
       'operator': '<' if bound=='min' else '>'
       }

    #######################
    # Post-ODE equations
    #######################
    if len(post_odes) > 0:
        code += """
        /////////////////////////
        // After the ODES
        /////////////////////////
"""
        for param in post_odes: 
            code += """
        %(comment)s
        %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }

            # Min-Max bounds
            for bound, val in param['bounds'].iteritems():
                code += """
        if(%(obj)s%(sep)s%(var)s%(index)s %(operator)s %(val)s)
            %(obj)s%(sep)s%(var)s%(index)s = %(val)s;
""" % {'obj': pattern['pop_prefix'] if obj == 'pop' else pattern['proj_prefix'],
       'sep': pattern['pop_sep'] if obj == 'pop' else pattern['proj_sep'],
       'index': pattern['pop_index'] if obj == 'pop' else pattern['proj_index'],
       'var' : param['name'], 'val' : val, 'id': id, 
       'operator': '<' if bound=='min' else '>'
       }

    return code
