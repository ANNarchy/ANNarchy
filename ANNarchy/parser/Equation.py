"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from .ParserTemplate import create_local_dict, user_functions

import sympy as sp
import re
from copy import deepcopy

def transform_condition(expr):
    """
    Transforms the "natural" logical operators into Sympy-compatible versions.
    """
    expr = expr.replace (' and ', ' & ')
    expr = expr.replace (' or ', ' | ')
    expr = expr.replace (' is not ', ' != ')
    expr = expr.replace (' not ', ' Not ')
    expr = expr.replace (' not(', ' Not(')
    expr = expr.replace (' is ', ' == ')

    return expr

class Equation(object):
    '''
    Class to analyse a single equation.
    '''
    def __init__(self,
                 name,
                 expression,
                 description,
                 untouched = [],
                 method='explicit',
                 type=None,
                 net_id=0,
        ):
        '''
        Parameters:

        * name : The name of the variable
        * expression: The expression as a string
        * variables: a list of all the variables in the neuron/synapse
        * local_variables: a list of the local variables
        * global_variables: a list of the global variables
        * method: the numerical method to use for ODEs
        * type: forces the analyser to consider the equation as: simple, cond, ODE, inc, otherwise it is guessed.
        * untouched: list of terms which should not be modified
        '''

        # Store attributes
        self.name = name
        self.expression = transform_condition(expression)
        self.description = description
        self.attributes = self.description['attributes']
        self.local_attributes = self.description['local']
        self.semiglobal_attributes = self.description['semiglobal']
        self.global_attributes = self.description['global']
        self.local_functions = [func['name'] for func in self.description['functions']]
        self.variables = [var['name'] for var in self.description['variables']]
        self.untouched = untouched
        self.method = method
        self.num_flops = 0
        self.net_id=net_id

        # Determine the type of the equation
        if not type:
            self.type = self.identify_type()
        else:
            self.type = type

        # Copy the list of built-in functions
        self.user_functions = user_functions.copy()

        # Add each user-defined function to avoid "not supported in C"
        for var in self.local_functions:
            self.user_functions[var] = var

        # Copy the default dictionary of built-in symbols or functions
        self.local_dict = create_local_dict(
            self.local_attributes,
            self.semiglobal_attributes,
            self.global_attributes,
            self.untouched,
            self.user_functions
            ) # in ParserTemplate


    def parse(self):
        "Main method called after creating the object."
        try:
            if self.type == 'ODE':
                code = self.analyse_ODE(self.expression)
            elif self.type == 'cond':
                code = self.analyse_condition(self.expression)
            elif self.type == 'inc':
                code = self.analyse_increment(self.expression)
            elif self.type == 'return':
                code = self.analyse_return(self.expression)
            elif self.type == 'simple':
                code = self.analyse_assignment(self.expression)
        except Exception as e:
            Messages._print(e)
            Messages._error('Parser: cannot analyse', self.expression)
        return code

    def identify_type(self):
        """
        Identifies which type has the equation:

        * inc for "a += 0.2"
        * ODE for "dV/dt + V = A"
        * cond for "mp > 30.0" or "r != 0.0"
        * simple for the rest, e.g. "r = pos(mp)" or "baseline = Uniform(0,1)"
        """
        # Suppress spaces to extract dvar/dt
        expression = self.expression.replace(' ', '')

        # Check if it an increment
        for op in ['+=', '-=', '*=', '/=']:
            if op in expression:
                return 'inc'

        # Check if it is an ode
        if 'd'+self.name+'/dt' in expression:
            return 'ODE'

        # Check if it is a condition (e.g spike)
        split_expr = expression.split('=')
        if len(split_expr) == 1: # no equal sign = treat as condition (return statements will be specified
            return 'cond'
        if split_expr[1] == '': # the first = is followed by another =
            return 'cond'
        for logop in ['<', '>', '!']: # the first operator is <=, >= or !=
            if logop == split_expr[0][-1]:
                return 'cond'

        # Only a simple assignement
        return 'simple'


    ###############################################
    ### Code generation
    ###############################################

    def c_code(self, equation):
        """
        Returns the C version of a Sympy expression

        Implementation note:

        Changes to this method should be applied also
        on ANNarchy.parser.CoupledEquations.c_code() too.
        """
        # The `strict` parameter has been introduced in sympy >= 1.13.
        # It defaults to `True`.
        try:
            c_code = sp.ccode(
                equation,
                precision=8,
                strict=False,
                user_functions=self.user_functions
            )
        except TypeError:
            c_code = sp.ccode(
                equation,
                precision=8,
                user_functions=self.user_functions
            )

        if ConfigManager().get('precision', self.net_id) == "float":
            #
            # Add the f-suffix to floating value constants
            matches = re.findall(r"[-]?[0-9]+\.[0-9]+", c_code)
            matches = sorted(list(set(matches)))    # remove doublons, e. g. 0.5*dt

            for m in matches:
                fval = float(m)
                fval = round(fval, 8)       # shorten the val to a reasonable length
                c_code = c_code.replace(m, str(fval)+"f")

            # If found constants have an overlap, e.g., 5.0 and 15.0, we create
            # multiple suffixes, e.g., 15.0ff. Therefore, we need to remove them
            # afterwards ...
            matches = re.findall(r"[-]?[0-9]+\.[0-9]+f[f]+", c_code)
            for m in matches:
                fval = float(m.split("f")[0])
                fval = round(fval, 8)       # shorten the val to a reasonable length
                c_code = c_code.replace(m, str(fval)+"f")

            # Replace the math functions with their single precision
            # to circumenvent problems induced by implicit type conversion
            # e. g. pow(double, double) by powf(float, float)
            for func in ["fabs", "pow", "exp", "sin", "cos", "tan", "round"]:
                c_code = c_code.replace(func+"(", func+"f(")

        return c_code

    def latex_code(self, equation):
        "Returns the LaTeX version of a Sympy expression"
        return sp.latex(equation)

    def parse_expression(self, expression, local_dict):
        " Parses a string with respect to the vocabulary defined in local_dict."
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number, rationalize
        try:
            res =  parse_expr(
                expression,
                local_dict = local_dict,
                transformations = (standard_transformations + (convert_xor, )),
                evaluate=False
            )
        except Exception as e:
            Messages._print(e)
            Messages._error('Can not analyse the expression :' +  str(expression))

        return res

    def _count_ops(self, equation):
        """
        Count the number of numeric operations to approximate the number of required
        floating point operations.
        """
        exp_symbol = sp.Symbol('EXP')

        # Some numeric operations are more expensive than others. For example,
        # Izhikevich (2004) states ten operations for exponential functions
        weighted_count = sp.count_ops(equation, visual=True).subs(exp_symbol, 10)

        # Every other operation gets a weight of 1 (the default)
        weighted_count = weighted_count.replace(sp.Symbol, type(sp.S.One))

        # return the (re-)weighted result
        return weighted_count

    ###############################################
    ### ODE
    ###############################################

    def analyse_ODE(self, expression):
        " Returns the C++ code corresponding to an ODE with the method defined in self.method"
        # transform the expression to suppress =
        if '=' in expression:
            expression = expression.replace('=', '- (')
            expression += ')'

        # Suppress spaces to extract dvar/dt
        expression = expression.replace(' ', '')

        if self.method == 'semiimplicit':
            return self.semiimplicit(expression)
        elif self.method == 'implicit':
            return self.implicit(expression)
        elif self.method == 'explicit':
            return self.explicit(expression)
        elif self.method == 'exponential':
            return self.exponential(expression)
        elif self.method == 'midpoint':
            return self.midpoint(expression)
        elif self.method == 'rk4':
            return self.runge_kutta_4(expression)
        elif self.method == 'event-driven':
            return self.eventdriven(expression)
        else:
            raise NotImplementedError("The chosen numerical method ", self.method, "is not available")

    def explicit(self, expression):
        " Explicit or backward Euler numerical method"

        expression = expression.replace('d'+self.name+'/dt', '_grad_var_')
        new_var = sp.Symbol('_grad_var_')
        self.local_dict['_grad_var_'] = new_var

        analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )

        self.analysed = analysed
        variable_name = self.local_dict[self.name]

        equation = sp.solve(analysed, new_var, check=False, rational=False)[0]
        equation = sp.simplify(equation, ratio=1.0)

        explicit_code = ConfigManager().get('precision', self.net_id) + ' _' + self.name + ' = ' + self.c_code(equation) + ';'

        switch = self.c_code(variable_name) + ' += dt*_' + self.name + ' ;'

        # compute required number of operations
        self.num_flops = self._count_ops(equation)    # compute increment
        self.num_flops += 2                          # apply increment

        # Return result
        return [{}, explicit_code, switch]


    def midpoint(self, expression):

        "Midpoint method."
        expression = expression.replace('d'+self.name+'/dt', '_grad_var_')
        new_var = sp.Symbol('_grad_var_')
        self.local_dict['_grad_var_'] = new_var

        analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )

        self.analysed = analysed

        variable_name = self.local_dict[self.name]

        # equation = sp.simplify(sp.collect( sp.solve(analysed, new_var, check=False)[0], self.local_dict['dt']))
        equation = sp.solve(analysed, new_var, check=False, rational=False)[0]
        equation = sp.collect(equation, self.local_dict['dt'])
        equation = sp.simplify(equation, ratio=1.0)

        explicit_code = ConfigManager().get('precision', self.net_id) + ' _k_' + self.name + ' = dt*(' + self.c_code(equation) + ');'
        # Midpoint method:
        # Replace the variable x by x+_x/2
        tmp_dict = self.local_dict
        tmp_dict[self.name] = sp.Symbol('(' + self.c_code(variable_name) + ' + 0.5*_k_' + self.name + ' )')
        tmp_analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )
        tmp_equation = sp.solve(tmp_analysed, new_var, check=False, rational=False)[0]
        explicit_code += '\n' + ConfigManager().get('precision', self.net_id) + ' _' + self.name + ' = ' + self.c_code(tmp_equation) + ';'

        switch = self.c_code(variable_name) + ' += dt*_' + self.name + ' ;'

        # compute required number of operations
        self.num_flops = self._count_ops(equation)        # compute increment (explicit)
        self.num_flops += self._count_ops(tmp_equation)   # compute increment (dt/2.0)
        self.num_flops += 2                              # apply increment

        # Return result
        return [{}, explicit_code, switch]

    def runge_kutta_4(self, expression):
        "Runge-Kutta 4th order method"

        expression = expression.replace('d'+self.name+'/dt', '_grad_var_')
        new_var = sp.Symbol('_grad_var_')
        self.local_dict['_grad_var_'] = new_var

        analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )

        self.analysed = analysed

        variable_name = self.local_dict[self.name]

        # k1 = f(x)
        equation = sp.solve(analysed, new_var, check=False, rational=False)[0]
        equation = sp.collect(equation, self.local_dict['dt'])
        equation = sp.simplify(equation, ratio=1.0)
        explicit_code = ConfigManager().get('precision', self.net_id) + ' _k1_' + self.name + ' = (' + self.c_code(equation) + ');\n'

        # k2 = f(x+dt/2*k)
        tmp_dict = deepcopy(self.local_dict)
        tmp_dict[self.name] = sp.Symbol('(' + self.c_code(variable_name) + ' + 0.5 * dt * _k1_' + self.name + ' )')
        tmp_analysed = self.parse_expression(expression,
            local_dict = tmp_dict
        )
        tmp_equation_k2 = sp.solve(tmp_analysed, new_var, check=False, rational=False)[0]
        explicit_code += ConfigManager().get('precision', self.net_id) + ' _k2_' + self.name + ' = (' + self.c_code(tmp_equation_k2) + ');\n'

        # k3 = f(x+dt/2*k2)
        tmp_dict = deepcopy(self.local_dict)
        tmp_dict[self.name] = sp.Symbol('(' + self.c_code(variable_name) + ' + 0.5 * dt * _k2_' + self.name + ' )')
        tmp_analysed = self.parse_expression(expression,
            local_dict = tmp_dict
        )
        tmp_equation_k3 = sp.solve(tmp_analysed, new_var, check=False, rational=False)[0]
        explicit_code += ConfigManager().get('precision', self.net_id) + ' _k3_' + self.name + ' = (' + self.c_code(tmp_equation_k3) + ');\n'

        # k4 = f(x+dt*k3)
        tmp_dict = deepcopy(self.local_dict)
        tmp_dict[self.name] = sp.Symbol('(' + self.c_code(variable_name) + ' + dt * _k3_' + self.name + ' )')
        tmp_analysed = self.parse_expression(expression,
            local_dict = tmp_dict
        )
        tmp_equation_k4 = sp.solve(tmp_analysed, new_var, check=False, rational=False)[0]
        explicit_code += ConfigManager().get('precision', self.net_id) + ' _k4_' + self.name + ' = (' + self.c_code(tmp_equation_k4) + ');\n'

        # final x is part of k1 .. k4
        switch = self.c_code(variable_name) + ' += dt/6.0 * ( _k1_' + self.name + ' + (_k2_' + self.name + '+_k2_' + self.name + ') + (_k3_' + self.name + '+_k3_' + self.name + ') + _k4_' + self.name + ');'

        # compute required number of operations
        self.num_flops = self._count_ops(equation)            # compute increment (explicit)
        self.num_flops += self._count_ops(tmp_equation_k2)    # compute increment (k2)
        self.num_flops += self._count_ops(tmp_equation_k3)    # compute increment (k3)
        self.num_flops += self._count_ops(tmp_equation_k4)    # compute increment (k4)
        self.num_flops += 8                                   # apply increment

        return [{}, explicit_code, switch]

    def implicit(self, expression):
        "Full implicit method, linearising for example (V - E)^2, but this is not desired."

        # print('Expression', expression)
        # Transform the gradient into a difference TODO: more robust...
        new_expression = expression.replace('d'+self.name, '_t_gradient_')
        new_expression = re.sub(r'([^\w]+)'+self.name+r'([^\w]+)', r'\1_'+self.name+r'\2', new_expression)
        new_expression = new_expression.replace('_t_gradient_', '(_'+self.name+' - '+self.name+')')

        # print('New Expression', new_expression)

        # Add a sympbol for the next value of the variable
        new_var = sp.Symbol('_'+self.name)
        self.local_dict['_'+self.name] = new_var

        # Parse the string
        analysed = self.parse_expression(
            new_expression,
            local_dict = self.local_dict
        )
        self.analysed = analysed
        # print('Analysed', analysed)

        # Solve the equation for delta_mp
        solved = sp.solve(analysed, new_var, check=False, rational=False)
        # print('Solved', solved)
        if len(solved) > 1:
            Messages._print(self.expression)
            Messages._error('Parser: the ODE is not linear, can not use the implicit method.')

        else:
            solved = solved[0]

        equation = sp.simplify(sp.collect( solved, self.local_dict['dt']))

        # Obtain C code
        variable_name = self.c_code(self.local_dict[self.name])

        explicit_code = ConfigManager().get('precision', self.net_id) + ' _' + self.name + ' = '\
                        +  self.c_code(equation) + ';'
        switch = variable_name + ' = _' + self.name + ' ;'

        # Return result
        return [{}, explicit_code, switch]

    def semiimplicit(self, expression):
        " Implicit or forward Euler numerical method, but only for the linear part of the equation."
        # Standardize the equation
        real_tau, stepsize, steadystate = self.standardize_ODE(expression)

        if real_tau is None: # the equation can not be standardized
            Messages._print(expression)
            Messages._warning('The implicit Euler method can not be applied to this equation (must be linear), applying explicit Euler instead.')
            return self.explicit(expression)

        instepsize = sp.together( stepsize / (stepsize + sp.S(1.0)) )

        # Obtain C code
        variable_name = self.c_code(self.local_dict[self.name])

        explicit_code = ConfigManager().get('precision', self.net_id) + ' _' + self.name + ' = ('\
                        +  self.c_code(instepsize) + ')*(' \
                        + self.c_code(steadystate)+ ' - ' + variable_name +');'
        switch = variable_name + ' += _' + self.name + ' ;'

        # Return result
        return [{}, explicit_code, switch]


    def exponential(self, expression):
        "Exponential Euler method."

        # Standardize the equation
        real_tau, stepsize, steadystate = self.standardize_ODE(expression)

        if real_tau is None: # the equation can not be standardized
            return self.explicit(expression)

        # Check if the step size is local or not
        global_stepsize = True
        for var in self.attributes: # Add each variable of the neuron
            if var in self.local_attributes and self.local_dict[var] in stepsize.atoms():
                global_stepsize = False

        # If the stepsize is global, take it out of the for loop to save some operations
        pre_loop = {}
        step_size_code = '(1.0 - exp(' + self.c_code(-stepsize) + '))'
        if global_stepsize:
            pre_loop['name'] = "__stepsize_" + self.name
            pre_loop['value'] = '1.0 - exp( ' + self.c_code(-stepsize) + ')'
            step_size_code = "__stepsize_" + self.name


        # Obtain C code
        variable_name = self.c_code(self.local_dict[self.name])

        explicit_code = ConfigManager().get('precision', self.net_id) + ' _' + self.name + ' =  ' \
                        + step_size_code \
                        + '*(' \
                        + self.c_code(steadystate) \
                        + ' - ' \
                        + self.c_code(self.local_dict[self.name]) \
                        +');'

        switch = variable_name + ' += _' + self.name + ' ;'

        # Return result
        return [pre_loop, explicit_code, switch]

    def eventdriven(self, expression):
        # Standardize the equation
        real_tau, _, steadystate = self.standardize_ODE(expression)

        if real_tau is None: # the equation can not be standardized
            Messages._print(self.expression)
            Messages._error('The equation is not a linear ODE and can not be evaluated exactly.')


        # Check the steady state is not dependent on other variables
        for var in self.variables:
            if self.local_dict[var] in steadystate.atoms():
                Messages._print(self.expression)
                Messages._error('The equation can not depend on other variables ('+var+') to be evaluated exactly.')


        # Obtain C code
        variable_name = self.c_code(self.local_dict[self.name])
        steady = self.c_code(steadystate)
        if steady == '0':
            code = variable_name + ' *= exp(dt*(_last_event%(local_index)s - (t))/(' + self.c_code(real_tau) + '));'
        else:
            code = variable_name + ' = ' + steady + ' + (' + variable_name + ' - ' + steady + ')*exp(dt*(_last_event%(local_index)s - (t))/(' + self.c_code(real_tau) + '));'
        return code

    def standardize_ODE(self, expression):
        """ Transform any 1rst order ODE into the standardized form:

        tau * dV/dt + V = S

        Non-linear functions of V are left in the steady-state argument.

        Returns:

            * tau : the time constant associated to the standardized equation.

            * stepsize: a simplified version of dt/tau.

            * steadystate: the right term of the equation after standardization
        """
        # Replace the gradient with a temporary variable
        expression = expression.replace('d' + self.name +'/dt', '_gradvar_') # TODO: robust to spaces

        # Add the gradient sympbol
        grad_var = sp.Symbol('_gradvar_')

        # Parse the string
        analysed = self.parse_expression(
            expression,
            local_dict = self.local_dict
        )
        self.analysed = analysed

        # Collect factor on the gradient and main variable A*dV/dt + B*V = C
        expanded = analysed.expand(
            modulus=None, power_base=False, power_exp=False,
            mul=True, log=False, multinomial=False)

        # Make sure the expansion went well
        collected_var = sp.collect(expanded, self.local_dict[self.name], evaluate=False, exact=False)

        if not self.local_dict[self.name] in collected_var.keys() or len(collected_var)>2:
            collected_var = sp.collect(sp.simplify(expanded), self.local_dict[self.name], evaluate=False, exact=False)
            if not self.local_dict[self.name] in collected_var.keys() or len(collected_var)>2:
                Messages._print(self.expression)
                print(collected_var)
                Messages._error('The exponential and event-driven methods are reserved for linear first-order ODEs of the type tau*d'+ self.name+'/dt + '+self.name+' = f(t). Use the explicit method instead.')

        factor_var = collected_var[self.local_dict[self.name]]

        collected_gradient = sp.collect(sp.expand(analysed, grad_var), grad_var, evaluate=False, exact=False)
        if grad_var in collected_gradient.keys():
            factor_gradient = collected_gradient[grad_var]
        else:
            factor_gradient = sp.S(1.0)

        # Real time constant when using the form tau*dV/dt + V = A
        real_tau = factor_gradient / factor_var

        # Normalized equation tau*dV/dt + V = A
        normalized = analysed / factor_var

        # Steady state A
        steadystate = sp.simplify(sp.together(real_tau * grad_var + self.local_dict[self.name] - normalized))

        # Stepsize
        stepsize = sp.simplify(sp.together(self.local_dict['dt']/real_tau))

        return real_tau, stepsize, steadystate

    ###############################################
    ### Conditionals
    ###############################################

    def analyse_condition(self, expression):
        " Analyzes a boolean condition (e.g. for the spike argument)."

        # print("Before:", expression, type(expression))

        expression = expression.strip()

        # Check if there is a == in the condition
        if '==' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)==([\w\s.]+)', r'Equality(\1, \2)', expression)
            else: # only one term
                # Remove brackets
                if expression[0] == "(" and expression[-1] == ")":
                    expression = expression[1:-1]
                terms = expression.split('==')
                expression = 'Equality(' + terms[0] + ', ' + terms[1] + ')'

        # Check if there is a != in the condition
        if '!=' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)!=([\w\s.]+)', r'Not(Equality(\1, \2))', expression)
            else:# only one term
                # Remove brackets
                if expression[0] == "(" and expression[-1] == ")":
                    expression = expression[1:-1]
                terms = expression.split('!=')
                expression = 'Not(Equality(' + terms[0] + ', ' + terms[1] + '))'

        # print("After", expression)

        # Parse the string
        analysed = self.parse_expression(
            expression,
            local_dict = self.local_dict
        )
        self.analysed = analysed
        # print(analysed)

        # Obtain C code
        code = self.c_code(analysed)
        # print(code)

        # Return result
        return code

    ###############################################
    ### Increments
    ###############################################

    def analyse_increment(self, expression):
        " Analyzes an incremental assignment (e.g. a += 0.2)."

        # Get only the right term
        if '+=' in expression:
            name = expression[:expression.find('+=')].strip()
            expression = expression[expression.find('+=')+2:]
            ope = ' += '
        elif '-=' in expression:
            name = expression[:expression.find('-=')].strip()
            expression = expression[expression.find('-=')+2:]
            ope = ' -= '
        elif '*=' in expression:
            name = expression[:expression.find('*=')].strip()
            expression = expression[expression.find('*=')+2:]
            ope = ' *= '
        elif '/=' in expression:
            name = expression[:expression.find('/=')].strip()
            expression = expression[expression.find('/=')+2:]
            ope = ' /= '

        # Parse the string
        analysed = self.parse_expression(
            expression,
            local_dict = self.local_dict
        )
        self.analysed = analysed

        # Obtain C code
        code = self.c_code(self.local_dict[name]) + ope + self.c_code(sp.simplify(analysed, ratio=1.0)) +';'

        # number of floating operations
        self.num_flops = 1 + self._count_ops(self.analysed)

        # Return result
        return code

    ###############################################
    ### Assignments
    ###############################################

    def analyse_assignment(self, expression):
        " Analyzes a simple assignment (e.g. a = 0.2)."

        # Get the left and right term
        name = expression[:expression.find('=')].strip()
        expression = expression[expression.find('=')+1:]

        # Parse the string
        self.analysed = self.parse_expression(
            expression,
            local_dict = self.local_dict
        )

        # Obtain C code
        code = self.c_code(self.local_dict[name]) + ' = ' + self.c_code(self.analysed) +';'

        # number of floating operations
        self.num_flops = self._count_ops(self.analysed)

        # Return result
        return code

    def analyse_return(self, expression):
        " Analyzes a return statement (e.g. w * pre.r)."

        # Parse the string
        analysed = self.parse_expression(
            expression,
            local_dict = self.local_dict
        )
        self.analysed = analysed

        # Obtain C code
        code = self.c_code(analysed) +';'

        # number of floating operations
        self.num_flops = self._count_ops(self.analysed)

        # Return result
        return code

    ###############################################
    ### Dependencies
    ###############################################

    def dependencies(self):
        "Returns all dependencies of the equation"
        deps = []
        for att in self.attributes:
            if self.local_dict[att] in self.analysed.atoms():
                deps.append(att)
        return deps
