"""

    Function.py

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
import ANNarchy.core.Global as Global
from ANNarchy.parser.Equation import transform_condition
from .ParserTemplate import parser_dict, functions_dict, user_functions

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number
import re

class FunctionParser(object):
    '''
    Class to analyse one equation.
    '''
    def __init__(self, 
                 name, 
                 expression, 
                 description,
                 untouched = [],
                 method='explicit', 
                 type=None):
        '''
        Parameters:

        * name : The name of the variable
        * expression: The expression as a string
        * variables: a list of all the variables in the neuron/synapse
        * local_variables: a list of the local variables
        * global_variables: a list of the global variables
        * method: the numerical method to use for ODEs
        * type: forces the analyser to consider the equation as: simple, cond, ODE, inc
        * untouched: list of terms which should not be modified
        '''

        self.args = description
        self.eq = expression

        # Copy the default functions dictionary
        self.local_dict = functions_dict.copy()
        # Add the arguments to the dictionary
        for arg in self.args:
            self.local_dict[arg] = sp.Symbol(arg)


        # Add custom constants
        for obj in Global._objects['constants']:
            self.local_dict[obj.name] = sp.Symbol(obj.name)

        # Add other functions    
        self.user_functions = user_functions.copy()
        for func in [func[0] for func in Global._objects['functions']]:
            self.user_functions[func] = func
            self.local_dict[func] = sp.Function(func)

        # Possibly conditionals (up to 10 per equation... dirty!)
        for i in range(10):
            self.local_dict['__conditional__'+str(i)] = sp.Symbol('__conditional__'+str(i))

    def parse(self, part=None):
        if not part:
            part = self.eq

        expression = transform_condition(part)

        # Check if there is a == in the condition
        if '==' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)==([\w\s.]+)', r'Equality(\1, \2)', expression)
            else:
                terms = expression.split('==')
                expression = 'Equality(' + terms[0] + ', ' + terms[1] + ')'

        # Check if there is a != in the condition
        if '!=' in expression:
            # Is it the only term, or are there other operations?
            if '&' in expression or '|' in expression:
                expression = re.sub(r'([\w\s.]+)!=([\w\s.]+)', r'Not(Equality(\1, \2))', expression)
            else:
                terms = expression.split('!=')
                expression = 'Not(Equality(' + terms[0] + ', ' + terms[1] + '))'

        try:
            eq = parse_expr(expression,
                local_dict = self.local_dict,
                transformations = ((auto_number, convert_xor,))
            )
        except:
            Global._print(expression)
            Global._error('The function depends on unknown variables.')

        return sp.ccode(eq, precision=8,
            user_functions=self.user_functions)

    def dependencies(self):
        "For compatibility with Equation."
        return self.args

