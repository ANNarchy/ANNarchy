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
from ANNarchy.core.Global import _warning
from ANNarchy.parser.Equation import transform_condition

from sympy import *
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, auto_number

class FunctionParser(object):
    '''
    Class to analyse one equation.
    '''
    def __init__(self, eq, args):
        self.args = args
        self.eq = eq
        self.local_dict = {
            'pos': Function('positive'),
            'positive': Function('positive'),
            'neg': Function('negative'),
            'negative': Function('negative'),
            'clip': Function('clip'),
            'True': Symbol('true'),
            'False': Symbol('false'),
        }
        for arg in self.args:
            self.local_dict[arg] = Symbol(arg)

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

        eq = parse_expr(expression,
            local_dict = self.local_dict,
            transformations = (standard_transformations + (convert_xor,))
        )
        return ccode(eq, precision=8)
