
def sort_odes(desc, locality='local'):
    pre_odes = []
    odes = []
    post_odes=[]
    is_pre_ode = True
    is_ode = False
    for param in desc['variables']: 
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

def generate_equation_code(desc, locality='local'):
    # Separate ODEs from the pre- and post- equations
    pre_odes, odes, post_odes = sort_odes(desc, locality)
    # Generate code
    code = ""  
    #########################       
    # Pre-ODE equations
    #########################
    for param in pre_odes: 
        code += """
    %(comment)s
    %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }
        # Min-Max bounds
        for bound, val in param['bounds'].iteritems():
            # Bound min
            if bound == 'min':
                code += """
    if(%(var)s_[i] < %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}
            # Bound max 
            if bound == 'max':
                code += """
    if(%(var)s_[i] > %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}

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
    if len(odes)>0:
        code += """
    // Switch values
    """
    for param in odes: 
        code += """
    %(comment)s
    %(switch)s 
""" % { 'comment': '// '+param['eq'],
        'switch' : param['switch']}
    
    # Min-Max bounds
    for param in odes: 
        for bound, val in param['bounds'].iteritems():
            # Bound min
            if bound == 'min':
                code += """
    %(comment)s
    if(%(var)s_[i] < %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'comment': '// '+param['eq'], 'var' : param['name'], 'val' : val}
            # Bound max 
            if bound == 'max':
                code += """
    %(comment)s
    if(%(var)s_[i] > %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'comment': '// '+param['eq'],'var' : param['name'], 'val' : val}

    #######################
    # Post-ODE equations
    #######################
    for param in post_odes: 
        code += """
    %(comment)s
    %(cpp)s
""" % { 'comment': '// '+param['eq'],
        'cpp': param['cpp'] }
        # Min-Max bounds
        for bound, val in param['bounds'].iteritems():
            # Bound min
            if bound == 'min':
                code += """
    if(%(var)s_[i] < %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}
            # Bound max 
            if bound == 'max':
                code += """
    if(%(var)s_[i] > %(val)s)
        %(var)s_[i] = %(val)s;
""" % {'var' : param['name'], 'val' : val}

    return code