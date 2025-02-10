"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

from .Equation import Equation
from .ParserTemplate import create_local_dict, user_functions

import sympy as sp

import re


class CoupledEquations(Equation):
    """
    Special equation solver when several equations are coupled and use the midpoint or implicit numerical methods.
    """

    def __init__(self, description, variables, net_id):
        # Global description
        self.description = description

        # List of equations to parse
        self.variables = variables

        # Net ID
        self.net_id = net_id

        # Build the list of expressions
        self.expression_list = {}
        for var in self.variables:
            self.expression_list[var['name']] = var['transformed_eq']

        # List of variable names to parse
        self.names = self.expression_list.keys()

        # Get attributes of the neuron/synapse
        self.local_attributes = self.description['local']
        self.semiglobal_attributes = self.description['semiglobal']
        self.global_attributes = self.description['global']
        self.untouched = variables[0]['untouched'] # the same for all eqs
        self.local_functions = [func['name'] for func in self.description['functions']]

        # Copy the default dictionary of built-in symbols or functions
        self.local_dict = create_local_dict(
            self.local_attributes, 
            self.semiglobal_attributes, 
            self.global_attributes, 
            self.untouched) # in ParserTemplate

        # Copy the list of built-in functions
        self.user_functions = user_functions.copy()
        # Add each user-defined function to avoid "not supported in C"
        for var in self.local_functions: 
            self.user_functions[var] = var

    def parse(self):
        "Main method called after creating the object."
        
        # Check if the numerical method is the same for all ODEs
        methods = []
        for var in self.variables:
            methods.append(var['method'])
        if len(list(set(methods))) > 1: # mixture of methods
            Messages._print(methods)
            Messages._error('Cannot mix different numerical methods when solving a coupled system of equations.')
            
        else:
            method = methods[0]

        if method == 'implicit' or method == 'semiimplicit':
            return self.solve_implicit(self.expression_list)
        elif method == 'midpoint': 
            return self.solve_midpoint(self.expression_list)
        elif method == 'rk4':
            return self.solve_rk4(self.expression_list)

    def solve_implicit(self, expression_list):
        "Implicit method"

        equations = {}
        new_vars = {}

        # Pre-processing to replace the gradient
        for name, expression in self.expression_list.items():
            # transform the expression to suppress =
            if '=' in expression:
                expression = expression.replace('=', '- (')
                expression += ')'
            # Suppress spaces to extract dvar/dt
            expression = expression.replace(' ', '')
            # Transform the gradient into a difference TODO: more robust...
            expression = expression.replace('d'+name, '_t_gradient_')
            expression_list[name] = expression

        # replace the variables by their future value
        for name, expression in expression_list.items():
            for n in self.names:
                expression = re.sub(r'([^\w]+)'+n+r'([^\w]+)', r'\1_'+n+r'\2', expression)
            expression = expression.replace('_t_gradient_', '(_'+name+' - '+name+')')
            expression_list[name] = expression

            new_var = sp.Symbol('_'+name)
            self.local_dict['_'+name] = new_var
            new_vars[new_var] = name

        for name, expression in expression_list.items():
            analysed = self.parse_expression(
                expression,
                local_dict = self.local_dict
            )
            equations[name] = analysed

        try:
            solution = sp.solve(list(equations.values()), list(new_vars.keys()))
        except Exception as e:
            Messages._print(expression_list)
            Messages._error('The multiple ODEs can not be solved together using the implicit Euler method.')     

        for var, sol in solution.items():
            # Simplify the solution
            sol  = sp.collect( sol, self.local_dict['dt'])

            # C++ code
            try:
                code = sp.ccode(sol, strict=False) 
            except:
                code = sp.ccode(sol) 


            # Generate the code
            cpp_eq = ConfigManager().get('precision', self.net_id) + ' _' + new_vars[var] + ' = ' + code + ';'
            switch = sp.ccode(self.local_dict[new_vars[var]] ) + ' = _' + new_vars[var] + ';'

            # Replace untouched variables with their original name
            for prev, new in sorted(list(self.untouched.items()), key = lambda key : len(key[0]), reverse=True):
                cpp_eq = re.sub(prev, new, cpp_eq)
                switch = re.sub(prev, new, switch)

            # Store the result
            for variable in self.variables:
                if variable['name'] == new_vars[var]:
                    variable['cpp'] = cpp_eq
                    variable['switch'] = switch

        return self.variables



    def solve_midpoint(self, expression_list):
        "Midpoint method"

        equations = {}
        evaluations = {}

        # Pre-processing to replace the gradient
        for name, expression in expression_list.items():
            
            # transform the expression to suppress =
            if '=' in expression:
                expression = expression.replace('=', '- (')
                expression += ')'
            
            # Suppress spaces to extract dvar/dt
            expression = expression.replace(' ', '')
            
            # Transform the gradient into a difference TODO: more robust..            
            expression = expression.replace('d'+name+'/dt', '_grad_var_'+name)
            new_var = sp.Symbol('_grad_var_'+name)
            self.local_dict['_grad_var_'+name] = new_var
            
            expression_list[name] = expression


        for name, expression in expression_list.items():

            analysed = self.parse_expression(expression,
                local_dict = self.local_dict
            )
            equations[name] = analysed
            evaluations[name] = sp.solve(analysed, self.local_dict['_grad_var_'+name])

        # Compute the k = f(x, t)
        ks = {}
        for name, evaluation in evaluations.items():
            ks[name] = ConfigManager().get('precision', self.net_id) + ' _k_' + name + ' = ' + self.c_code(evaluation[0]) + ';'

        # New dictionary replacing x by x+dt/2*k)
        tmp_dict = {}
        for name, val in self.local_dict.items():
            tmp_dict[name] = val
        for name, evaluation in evaluations.items():
            tmp_dict[name] = sp.Symbol('(' + sp.ccode(self.local_dict[name]) + ' + 0.5 * dt * _k_' + name + ' )')

        # Compute the new values _x_new = f(x + dt/2*_k)
        news = {}
        for name, expression in expression_list.items():
            tmp_analysed = self.parse_expression(expression,
                local_dict = tmp_dict
            )
            solved = sp.solve(tmp_analysed, self.local_dict['_grad_var_'+name])
            news[name] = ConfigManager().get('precision', self.net_id) + ' _' + name + ' = ' + self.c_code(solved[0]) + ';'

        # Compute the switches
        switches = {}
        for name, expression in expression_list.items():
            switches[name] = sp.ccode(self.local_dict[name]) + ' += dt * _' + name + ' ;'

        # Store the generated code in the variables
        for name in self.names:
            k = ks[name]
            n = news[name]
            switch = switches[name]

            # Replace untouched variables with their original name
            for prev, new in sorted(list(self.untouched.items()), key = lambda key : len(key[0]), reverse=True):
                k = re.sub(prev, new, k)
                n = re.sub(prev, new, n)
                switch = re.sub(prev, new, switch)

            # Store the result
            for variable in self.variables:
                if variable['name'] == name:
                    variable['cpp'] = [k, n]
                    variable['switch'] = switch
            
        return self.variables

    def solve_rk4(self, expression_list):
        "Runge-Kutta 4th order"

        expression_list = {}
        equations = {}
        evaluations = {}

        # Pre-processing to replace the gradient
        for name, expression in self.expression_list.items():
            # transform the expression to suppress =
            if '=' in expression:
                expression = expression.replace('=', '- (')
                expression += ')'
            # Suppress spaces to extract dvar/dt
            expression = expression.replace(' ', '')
            # Transform the gradient into a difference TODO: more robust...
            expression = expression.replace('d'+name+'/dt', '_gradient_'+name)
            self.local_dict['_gradient_'+name] = sp.Symbol('_gradient_'+name)
            expression_list[name] = expression

        for name, expression in expression_list.items():
            analysed = self.parse_expression(expression,
                local_dict = self.local_dict
            )
            equations[name] = analysed
            evaluations[name] = sp.solve(analysed, self.local_dict['_gradient_'+name])

        # Compute the k1 = f(x, t)
        k1_dict = {}
        for name, evaluation in evaluations.items():
            k1_dict[name] = ConfigManager().get('precision', self.net_id) + ' _k1_' + name + ' = ' + self.c_code(evaluation[0]) + ';'

        # New dictionary replacing x by x+dt/2*k1)
        k2_dict = {}
        tmp_dict_k2 = {}
        for name, val in self.local_dict.items():
            tmp_dict_k2[name] = val
        for name, evaluation in evaluations.items():
            tmp_dict_k2[name] = sp.Symbol('(' + sp.ccode(self.local_dict[name]) + ' + 0.5 * dt * _k1_' + name + ' )')

        # Compute the values _k2_x = f(x + dt/2*_k1)
        for name, expression in expression_list.items():
            tmp_analysed = self.parse_expression(expression,
                local_dict = tmp_dict_k2
            )

            solved = sp.solve(tmp_analysed, self.local_dict['_gradient_'+name])
            k2_dict[name] = ConfigManager().get('precision', self.net_id) + ' _k2_' + name + ' = ' + self.c_code(solved[0]) + ';'

        # New dictionary replacing x by x+dt/2*k2)
        k3_dict = {}
        tmp_dict_k3 = {}
        for name, val in self.local_dict.items():
            tmp_dict_k3[name] = val
        for name, evaluation in evaluations.items():
            tmp_dict_k3[name] = sp.Symbol('(' + sp.ccode(self.local_dict[name]) + ' + 0.5 * dt *_k2_' + name + ' )')

        # Compute the values _k3_x = f(x + dt/2*_k2_x)
        for name, expression in expression_list.items():
            tmp_analysed = self.parse_expression(expression,
                local_dict = tmp_dict_k3
            )

            solved = sp.solve(tmp_analysed, self.local_dict['_gradient_'+name])
            k3_dict[name] = ConfigManager().get('precision', self.net_id) + ' _k3_' + name + ' = ' + self.c_code(solved[0]) + ';'

        # New dictionary replacing x by x+dt*k3)
        k4_dict = {}
        tmp_dict_k4 = {}
        for name, val in self.local_dict.items():
            tmp_dict_k4[name] = val
        for name, evaluation in evaluations.items():
            tmp_dict_k4[name] = sp.Symbol('(' + sp.ccode(self.local_dict[name]) + ' + dt*_k3_' + name + ' )')

        # Compute the values _k4_x = f(x + dt*_k3_x)
        for name, expression in expression_list.items():
            tmp_analysed = self.parse_expression(expression,
                local_dict = tmp_dict_k4
            )

            solved = sp.solve(tmp_analysed, self.local_dict['_gradient_'+name])
            k4_dict[name] = ConfigManager().get('precision', self.net_id) + ' _k4_' + name + ' = ' + self.c_code(solved[0]) + ';'

        # accumulate _k1 - _k4 within the switch step
        dt_code = "dt/6.0f" if ConfigManager().get('precision', self.net_id) == 'float' else "dt/6.0"
        switches = {}
        for name, expression in expression_list.items():
            switches[name] = sp.ccode(self.local_dict[name]) + ' += '+dt_code+' * (_k1_'+name+' + (_k2_'+name+' + _k2_'+name+') + (_k3_'+name+' + _k3_'+name+') + _k4_'+name+');'

        # Store the generated code in the variables
        for name in self.names:
            k1 = k1_dict[name]
            k2 = k2_dict[name]
            k3 = k3_dict[name]
            k4 = k4_dict[name]
            switch = switches[name]

            # Replace untouched variables with their original name
            for prev, new in sorted(list(self.untouched.items()), key = lambda key : len(key[0]), reverse=True):
                k1 = re.sub(prev, new, k1)
                k2 = re.sub(prev, new, k2)
                k3 = re.sub(prev, new, k3)
                k4 = re.sub(prev, new, k4)
                switch = re.sub(prev, new, switch)

            # Store the result
            for variable in self.variables:
                if variable['name'] == name:
                    variable['cpp'] = [k1, k2, k3, k4]
                    variable['switch'] = switch

        return self.variables