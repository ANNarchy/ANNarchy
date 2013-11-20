"""

    Tree.py

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
from Definitions import *
import Nodes
import Analyser
import copy

# State machine for a single expression
class Tree(object):

    def __init__(self, analyser, name, expr):

        self.analyser = analyser
        self.targets = self.analyser.targets
        self.name = name
        self.expr = expr

        self.parameters = self.analyser.parameters_names
        self.other_variables = []
        for n in self.analyser.variables_names:
            if not n == self.name:
                self.other_variables.append(n)
        self.other_variables.append('t') # time can be used by default

        # Synapse analysers may need value or psp without defining it
        if isinstance(self.analyser, Analyser.SynapseAnalyser):
            if not 'psp' in self.other_variables:
                if not self.name == 'psp':
                    self.other_variables.append('psp')
            if not 'value' in self.other_variables:
                if not self.name == 'value':
                    self.other_variables.append('value')

        # Perform the analysis
        self.success = self.analyse()


    def analyse(self):
        # Expand all whitespaces
        self.expr_list = self.expand_whitespaces(self.expr).split(' ')
        while '' in self.expr_list:
            self.expr_list.remove('')
        # Check for an equal sign
        if not EQUAL in self.expr_list:
            print 'Error in the evaluation of', self.name, '(', self.expr, '): there is no equal sign.'
            return False
        # Start the recursive analysis
        self.tree = Nodes.TreeBuilder(self, self.expr_list).build()
        if not self.tree:
            return False
        # Move the tree so that only the target variable is on the left.
        # Search for the gradient on the left side if any
        if lookup(self.tree.left, Nodes.Gradient) != None:
            self.simplifiedtree = simplify(copy.copy(self.tree), Nodes.Gradient)
            self.simplifiedtree.order = 1
        elif lookup(self.tree.left, Nodes.MainVariable) != None:
            self.simplifiedtree = simplify(copy.copy(self.tree), Nodes.MainVariable)
            self.simplifiedtree.order = 0
        else:
            return False
        return True

    def expand_whitespaces(self, expr):
        # remove carriage returns
        expr=expr.replace('\n', ' ')
        # Split arounf the operators
        for op in OPERATORS:
            expr=expr.replace(op, ' '+op+' ')
        for op in BRACKETS:
            expr=expr.replace(op, ' '+op+' ')
        # COMPARATORS:
        import re
        expr=expr.replace('<=', ' <= ')
        expr=expr.replace('>=', ' >= ')
        expr=expr.replace('!=', ' != ')
        expr=expr.replace('==', ' == ')
        expr = re.sub('<(?P<post>[^=])', ' < \g<post>', expr) # Avoid splitting the comparators
        expr = re.sub('>(?P<post>[^=])', ' > \g<post>', expr) # Avoid splitting the comparators
        code = re.split('([^<>!=])=([^=])', expr) # Avoid splitting the comparators
        if len(code) != 4:
            print code
            print 'Error while analysing', expr, ': There should be only one equal sign in the equation.'
            exit(0)
        expr=code[0] + code[1] + ' = ' + code[2] + code[3]
        return expr

    def cpp(self):
        return self.simplifiedtree.cpp()

    def latex(self, simplified=False):
        if simplified:
            return self.simplifiedtree.latex()
        else:
            return self.tree.latex()


## Utility functions
def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

# Depth-first search, tagging nodes on the way up
def lookup(obj, Type):
    if obj == None:
        obj.has_target=False
        return None
    if isinstance(obj, Type):
        obj.has_target=True
        return obj
    if hasattr(obj, 'left'):
        child=lookup(obj.left, Type)
        if isinstance(child, Type):
            obj.has_target=True
            return child
        else:
            obj.has_target=False
    if hasattr(obj, 'right'):
        child=lookup(obj.right, Type)
        if isinstance(child, Type):
            obj.has_target=True
            return child
        else:
            obj.has_target=False
    obj.has_target=False
    return None

#modify the left term so that only the target is left
def simplify(tree, target):
    if isinstance(tree.left, target):
        return tree
    if isinstance(tree.left, Nodes.PlusOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = Nodes.MinusOperator(tree.machine,
                                             left=tree.right,
                                             right=Nodes.Parenthesis(tree.machine, value=None, child=tree.left.right, hierarchize=False),
                                             hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = Nodes.MinusOperator(tree.machine,
                                             left=tree.right,
                                             right=Nodes.Parenthesis(tree.machine, value=None, child=tree.left.left, hierarchize=False),
                                             hierarchize=False)
            tree.left = tree.left.right
    elif isinstance(tree.left, Nodes.MinusOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = Nodes.PlusOperator(tree.machine,
                                            left=tree.right,
                                            right=tree.left.right,
                                            #right=Nodes.Parenthesis(tree.machine, value=None, child=tree.left.right, hierarchize=False),
                                            hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = Nodes.MinusOperator(tree.machine,
                                             left=tree.left.left,
                                             #left=Nodes.Parenthesis(tree.machine, value=None, child=tree.left.left, hierarchize=False),
                                             right=tree.right,
                                             hierarchize=False)
            tree.left = tree.left.right

    elif isinstance(tree.left, Nodes.MultOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = Nodes.DivOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = Nodes.DivOperator(tree.machine, left=tree.right, right=tree.left.left, hierarchize=False)
            tree.left = tree.left.right

    elif isinstance(tree.left, Nodes.DivOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = Nodes.MultOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = Nodes.DivOperator(tree.machine, left=tree.left.left, right=tree.right, hierarchize=False)
            tree.left = tree.left.right

    elif isinstance(tree.left, Nodes.PowOperator):
        if tree.left.left.has_target: # the left branch has the target
            if not float(tree.left.right.value) == 2.0:
                tree.right = Nodes.PowOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
                tree.right.right.value = str(1.0/float(tree.right.right.value))
                tree.left = tree.left.left
            else:
                tree.right = Nodes.Function(tree.machine, value = 'sqrt', child=tree.right, hierarchize=False)
                tree.left = tree.left.left
        else:
            print 'Error: can not solve a^x = b yet.'
            exit(0)

    else:
        print 'Error: one node is not invertible in the left member.'
        exit(0)

    return simplify(tree, target)

