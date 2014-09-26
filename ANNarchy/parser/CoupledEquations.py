"""

    CoupledEquations.py

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
from ANNarchy.core.Global import _warning, _error
from ANNarchy.parser.Equation import Equation

from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number

import re, pprint


class CoupledEquations(object):

    def __init__(self, pop, variables):
        self.pop = pop
        self.variables = variables

        self.untouched = variables[0]['untouched']

        self.expressions = {}
        for var in self.variables:
            self.expressions[var['name']] = var['transformed_eq']


        self.names = self.expressions.keys()

        self.local_variables = self.pop.description['local']
        self.global_variables = self.pop.description['global']

        self.local_dict = Equation('tmp', '',
                                      self.pop.description['attributes'], 
                                      self.pop.description['local'], 
                                      self.pop.description['global'], 
                                      method = 'implicit',
                                      untouched = self.untouched
                                      ).local_dict


    def process_variables(self):

        # Check if the numerical method is the same for all ODEs
        methods = []
        for var in self.variables:
            methods.append(var['method'])
        if len(list(set(methods))) > 1: # mixture of methods
            _error('Can not mix different numerical methods when solving a system of equations.')
            return None
        else:
            method = methods[0]

        if method == 'implicit' or method == 'semiimplicit':
            return self.solve_implicit(expressions)
        elif method == 'midpoint': 
            return self.solve_midpoint(expressions)


    def solve_implicit(self, expressions):

        equations = {}
        new_vars = {}

        # Pre-processing to replace the gradient
        for name, expression in self.expressions.iteritems():
            # transform the expression to suppress =
            if '=' in expression:
                expression = expression.replace('=', '- (')
                expression += ')'
            # Suppress spaces to extract dvar/dt
            expression = expression.replace(' ', '')
            # Transform the gradient into a difference TODO: more robust...
            expression = expression.replace('d'+name, '_t_gradient_')
            expressions[name] = expression

        # replace the variables by their future value
        for name, expression in expressions.iteritems():
            for n in names:
                expression = expression.replace(n, '_'+n)
            expression = expression.replace('_t_gradient_', '(_'+name+' - '+name+')')
            expressions[name] = expression

            new_var = Symbol('_'+name)
            self.local_dict['_'+name] = new_var
            new_vars[new_var] = name


        for name, expression in expressions.iteritems():
            analysed = parse_expr(expression,
                local_dict = self.local_dict,
                transformations = (standard_transformations + (convert_xor,))
            )
            equations[name] = analysed

            
        try:
            solution = solve(equations.values(), new_vars.keys())
        except:
            _error('The multiple ODEs can not be solved together using the implicit Euler method. Using the implicit Euler method for each ODE separately.')
            return None

        for var, sol in solution.iteritems():
            # simplify the solution
            sol  = collect( sol, self.local_dict['dt'])

            # Generate the code
            cpp_eq = 'double _' + new_vars[var] + ' = ' + ccode(sol) + ';'
            switch =  ccode(self.local_dict[new_vars[var]] ) + ' = _' + new_vars[var] + ';'

            # Replace untouched variables with their original name
            for prev, new in self.untouched.iteritems():
                cpp_eq = re.sub(prev, new, cpp_eq)
                switch = re.sub(prev, new, switch)

            # Store the result
            for variable in self.variables:
                if variable['name'] == new_vars[var]:
                    variable['cpp'] = cpp_eq
                    variable['switch'] = switch
            
        return self.variables



    def solve_midpoint(self, expressions):

        expressions = {}
        equations = {}
        evaluations = {}

        # Pre-processing to replace the gradient
        for name, expression in self.expressions.iteritems():
            # transform the expression to suppress =
            if '=' in expression:
                expression = expression.replace('=', '- (')
                expression += ')'
            # Suppress spaces to extract dvar/dt
            expression = expression.replace(' ', '')
            # Transform the gradient into a difference TODO: more robust...
            expression = expression.replace('d'+name+'/dt', '_gradient_'+name)
            self.local_dict['_gradient_'+name] = Symbol('_gradient_'+name)
            expressions[name] = expression

        for name, expression in expressions.iteritems():
            analysed = parse_expr(expression,
                local_dict = self.local_dict,
                transformations = (standard_transformations + (convert_xor,))
            )
            equations[name] = analysed
            evaluations[name] = solve(analysed, self.local_dict['_gradient_'+name])


        # Compute the k = f(x, t)
        ks = {}
        for name, evaluation in evaluations.iteritems():
            ks[name] = 'double _k_' + name + ' = ' + ccode(evaluation[0]) + ';'

        # New dictionary replacing x by x+dt/2*k)
        tmp_dict = {}
        for name, val in self.local_dict.iteritems():
            tmp_dict[name] = val
        for name, evaluation in evaluations.iteritems():
            tmp_dict[name] = Symbol('(' + ccode(self.local_dict[name]) + ' + 0.5*dt*_k_' + name + ' )')

        # Compute the new values _x_new = f(x + dt/2*_k)
        news = {}
        for name, expression in expressions.iteritems():
            tmp_analysed = parse_expr(expression,
                local_dict = tmp_dict,
                transformations = (standard_transformations + (convert_xor,))
            )
            solved = solve(tmp_analysed, self.local_dict['_gradient_'+name])
            news[name] = 'double _' + name + ' = ' + ccode(solved[0]) + ';'

        # Compute the switches
        switches = {}
        for name, expression in expressions.iteritems():
            switches[name] = ccode(self.local_dict[name]) + ' += dt * _' + name + ' ;'

        # Store the generated code in the variables
        for name in self.names:
            k = ks[name]
            n = news[name]
            switch = switches[name]

            # Replace untouched variables with their original name
            for prev, new in self.untouched.iteritems():
                k = re.sub(prev, new, k)
                n = re.sub(prev, new, n)
                switch = re.sub(prev, new, switch)

            # Store the result
            for variable in self.variables:
                if variable['name'] == name:
                    variable['cpp'] = [k, n]
                    variable['switch'] = switch
            
        return self.variables


        expression = expression.replace('d'+self.name+'/dt', '_grad_var_')
        new_var = Symbol('_grad_var_')
        self.local_dict['_grad_var_'] = new_var

        analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )


        variable_name = self.local_dict[self.name]

        equation = simplify(collect( solve(analysed, new_var)[0], self.local_dict['dt']))

        explicit_code =  'double _k_' + self.name + ' = dt*(' + self.c_code(equation) + ');'

        # Midpoint method:
        # Replace the variable x by x+_x/2
        tmp_dict = self.local_dict
        tmp_dict[self.name] = Symbol('(' + self.c_code(variable_name) + ' + 0.5*_k_' + self.name + ' )')
        tmp_analysed = self.parse_expression(expression,
            local_dict = self.local_dict
        )
        tmp_equation = solve(tmp_analysed, new_var)[0]

        explicit_code += '\n    double _' + self.name + ' = ' + self.c_code(tmp_equation) + ';'

        switch = self.c_code(variable_name) + ' += dt*_' + self.name + ' ;'

        # Return result
        return [explicit_code, switch]