"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Authorized keywork for attributes
authorized_keywords = [
    # Init
    'init',
    # Bounds
    'min',
    'max',
    # Locality
    'population',
    'postsynaptic',
    'projection',
    # Numerical methods
    'explicit',
    'implicit',
    'semiimplicit',
    'exponential',
    'midpoint',
    'rk4',
    'runge-kutta4', # backward compatibility
    'exact',
    'event-driven',
    # Refractory
    'unless_refractory',
    # Type
    'int',
    'bool',
    'float',
    # Event-based
    'unless_post',
]

####################################
# Functions for string manipulation
####################################
import re
from ANNarchy.intern import Messages

def split_equation(definition):
    " Splits a description into equation and flags."
    try:
        equation, constraint = definition.rsplit(':', 1)
    except ValueError:
        equation = definition.strip() # there are no constraints
        constraint = None
    else:
        has_constraint = False
        for keyword in authorized_keywords:
            if keyword in constraint:
                has_constraint = True
        if has_constraint:
            equation = equation.strip()
            constraint = constraint.strip()
        else:
            equation = definition.strip() # there are no constraints
            constraint = None
    finally:
        return equation, constraint

def prepare_string(stream):
    """ 
    Splits up a multiline equation, remove comments and unneeded spaces or tabs.
    """
    expr_set = []
    # replace ; with new line and split the result up
    tmp_set = stream.replace(';', '\n').split('\n')
    for expr in tmp_set:
        expr = re.sub(r'\#[\s\S]+', ' ', expr)   # remove comments
        expr = re.sub(r'\s+', ' ', expr)     # remove additional tabs etc.
        if expr.strip() == '' or len(expr)==0: # through beginning line breaks or something similar empty strings are contained in the set
            continue
        expr_set.append(''.join(expr))
    return expr_set

def extract_name(equation, left=False):
    " Extracts the name of a parameter/variable by looking the left term of an equation."
    equation = equation.replace(' ','')
    if not left: # there is potentially an equal sign
        try:
            name = equation.split('=')[0]
        except: # No equal sign. Eg: baseline : init=0.0
            return equation.strip()

        # Search for increments
        operators = ['+=', '-=', '*=', '/=', '>=', '<=']
        for op in operators:
            if op in equation:
                return equation.split(op)[0]
    else:
        name = equation.strip()
        # Search for increments
        operators = ['+', '-', '*', '/']
        for op in operators:
            if equation.endswith(op):
                return equation.split(op)[0]
    # Check for error
    if name.strip() == "":
        Messages._error('the variable name can not be extracted from ' + equation)

    # Search for any operation in the left side
    operators = ['+', '-', '*', '/']
    ode = False
    for op in operators:
        if not name.find(op) == -1:
            ode = True
    if not ode: # variable name is alone on the left side
        return name
    # ODE: the variable name is between d and /dt
    name = re.findall(r"d([\w]+)/dt", name)
    if len(name) == 1:
        return name[0].strip()
    else:
        return '_undefined'


def extract_flags(constraint):
    """
    Extracts from all attributes given after : which are bounds (eg min=0.0 or init=0.1) and which are flags (eg postsynaptic, implicit...).
    """

    bounds = {}
    flags = []

    # Check if there are constraints at all
    if not constraint:
        return bounds, flags
    
    # Regexp explanation:
    # (?:[^,()]+|\((?:[^()]*)\))+  # Matches either:
    #   - Text not containing commas or parentheses
    #   - Parenthesized content (balanced)
    pattern = r'(?:[^,()]+|\([^()]*\))+' 
    
    for con in [param.strip() for param in re.findall(pattern, constraint)]:
        try: # bound of the form key = val
            key, value = con.split('=')
            bounds[key.strip()] = value.strip()
        except ValueError: # No equal sign = flag
            flags.append(con.strip())

    return bounds, flags


def process_equations(equations):
    """ 
    Takes a multi-string describing equations and returns a list of dictionaries, where:

    * 'name' is the name of the variable
    * 'eq' is the equation
    * 'constraints' is all the constraints given after the last :. _extract_flags() should be called on it.
    """
    def is_constraint(eq):
        " Internal method to determine if a string contains reserved keywords."
        eq = ',' +  eq.replace(' ', '') + ','
        for key in authorized_keywords:
            pattern = '([,]+)' + key + '([=,]+)'
            if re.match(pattern, eq):
                return True
        return False

    # All equations will be stored there, in the order of their definition
    variables = []
    try:
        equations = equations.replace(';', '\n').split('\n')
    except: # equations is empty
        return variables


    # Iterate over all lines
    for line in equations:
        # Skip empty lines
        definition = line.strip()
        if definition == '':
            continue

        # Remove comments
        com = definition.split('#')
        if len(com) > 1:
            definition = com[0]
            if definition.strip() == '':
                continue

        # Process the line
        try:
            equation, constraint = definition.rsplit(':', 1)
        
        except ValueError: # There is no :, only equation is concerned
            equation = definition
            constraint = ''
        
        else:   # there is a :
            # Check if the constraint contains the reserved keywords
            has_constraint = is_constraint(constraint)
        
            # If the right part of : is a constraint, just store it
            # Otherwise, it is an if-then-else statement
            if has_constraint:
                equation = equation.strip()
                constraint = constraint.strip()
            else:
                equation = definition.strip() # there are no constraints
                constraint = ''

        # Split the equation around operators = += -= *= /=, but not ==
        split_operators = re.findall(r'([\s\w\+\-\*\/\)]+)=([^=])', equation)
        if len(split_operators) == 1: # definition of a new variable
            # Retrieve the name
            eq = split_operators[0][0]
            if eq.strip() == "":
                Messages._print(equation)
                Messages._error('The equation can not be analysed, check the syntax.')

            name = extract_name(eq, left=True)
            if name in ['_undefined', '']:
                Messages._error('No variable name can be found in ' + equation)

            # Append the result
            variables.append({'name': name, 'eq': equation.strip(), 'constraint': constraint.strip()})
        elif len(split_operators) == 0:
            # Continuation of the equation on a new line: append the equation to the previous variable
            variables[-1]['eq'] += ' ' + equation.strip()
            variables[-1]['constraint'] += constraint
        else:
            Messages._error('Only one assignement operator is allowed per equation.')

    return variables
