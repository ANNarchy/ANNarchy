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

        names = self.expressions.keys()
        expressions = {}
        equations = {}

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


        new_vars = {}
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
            _error('the multiple ODEs can not be solved together using the implicit Euler method. Using the Euler method for each ODE separately.')
            return None

        for var, sol in solution.iteritems():
            # simplify the solution
            sol  = collect( sol, self.local_dict['dt'])

            # Generate the code
            cpp_eq = 'DATA_TYPE _' + new_vars[var] + ' = ' + ccode(sol) + ';'
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