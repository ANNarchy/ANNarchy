"""

    Group.py
    
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
import Tree
        
# Group of expressions with parenthesis                
class Group(Nodes.Node):

    def __init__(self, machine, items):
        Nodes.Node.__init__(self, machine)
        self.machine = machine
        self.items = items
        
    def __repr__(self):
        return '(' + str(self.items) + ')'
        
    def cpp(self): # useless?
        return '(' + str(self.items) + ')'
        
    def latex(self):
        return str(self.items)
        
    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, k):
        return self.items[k]
        
    def group(self):
        complete = False
        while not complete:
            rk = 0
            idx_left = idx_right = 0
            complete=True
            # Find first-level brackets
            for idx in range(len(self.items)):
                it = self.items[idx]
                if it == BRACKET_LEFT:
                    complete=False
                    rk += 1
                    if rk == 1: # first bracket
                        idx_left = idx
                elif it == BRACKET_RIGHT:
                    rk -= 1
                    if rk== 0: # found the last matching bracket
                        idx_right = idx
                        break
            # Expand
            if idx_right != idx_left:
                subgroup = Group(self.machine, self.items[idx_left+1:idx_right])
                items=[]
                for i in range(idx_left):
                    items.append(self.items[i])
                items.append(subgroup)
                for i in range(idx_right+1, len(self.items)):
                    items.append(self.items[i])
                subgroup.group()
                self.items = items
            
    def analyse(self):
        items=[]
        # Turn the items into objects
        iter_items = iter(self.items)
        for item in iter_items:
            found = False
            if isinstance(item, str):
                # Check if it is a constant
                if belongs_to(item, CONSTANTS):
                    items.append(Nodes.Constant(self.machine, item))
                    found = True
                # Check if it is an integer or float
                if Tree.isFloat(item) or Tree.isInt(item):
                    items.append(Nodes.Constant(self.machine, item))
                    found=True
                # Check if it is a parameter
                if item in self.machine.parameters:
                    items.append(Nodes.Parameter(self.machine, item))
                    found=True
                # Check if it is a variable
                if item in self.machine.other_variables:
                    items.append(Nodes.Variable(self.machine, item))
                    found=True
                # Check if it is a global function (min, max, mean): pass the child as if
                if belongs_to(item, GLOBAL_FUNCTIONS):
                    items.append(Nodes.GlobalFunction(self.machine, value=item, child = iter_items.next()))
                    found = True
                # Check if it is a function
                if belongs_to(item, FUNCTIONS):
                    items.append(Nodes.Function(self.machine, item))
                    found = True
                # Check if it is a ternary operator
                if belongs_to(item, IF):
                    items.append(Nodes.If(self.machine, item))
                    found = True
                if belongs_to(item, THEN):
                    items.append(Nodes.Then(self.machine, item))
                    found = True
                if belongs_to(item, ELSE):
                    items.append(Nodes.Else(self.machine, item))
                    found = True
                # Check if it is a weighted sum
                if belongs_to(item, SUMS):
                    items.append(Nodes.PSP(self.machine, iter_items.next()))
                    found = True
                # Check if it is a pre variable
                if not item.find('pre.') == -1:
                    items.append(Nodes.PreVariable(self.machine, item))
                    found = True
                # Check if it is a post variable
                if not item.find('post.') == -1:
                    items.append(Nodes.PostVariable(self.machine, item))
                    found = True
                # Check if it is the updated variable
                if item == self.machine.name:
                    items.append(Nodes.MainVariable(self.machine, item))
                    found=True
                if item == 'd' + self.machine.name: # found the first item of a gradient
                    items.append(Nodes.Gradient(self.machine, self.machine.name))
                    if not iter_items.next() == '/' or not iter_items.next() == 'dt':
                        print 'Error: the gradient on', self.machine.name, 'is badly expressed, it should be', 'd'+self.machine.name+'/dt'
                        exit(0)
                    found=True
                # Check if it is an operator
                if belongs_to(item, PLUS):
                    items.append(Nodes.PlusOperator(self.machine, item))
                    found=True
                if belongs_to(item, MINUS):
                    items.append(Nodes.MinusOperator(self.machine, item))
                    found=True
                if belongs_to(item, MULT):
                    items.append(Nodes.MultOperator(self.machine, item))
                    found=True
                if belongs_to(item, DIV):
                    items.append(Nodes.DivOperator(self.machine, item))
                    found=True  
                if belongs_to(item, POW):
                    items.append(Nodes.PowOperator(self.machine, item))
                    found=True  
                # Check if it is a comparator
                if belongs_to(item, COMPARATORS):
                    items.append(Nodes.Comparator(self.machine, item))
                    found = True  
                # Check if it is a logical operator
                if belongs_to(item, LOGICALS):
                    items.append(Nodes.Logical(self.machine, item))
                    found = True  
                # Check if it is a not operator
                if belongs_to(item, NOT):
                    items.append(Nodes.Not(self.machine, item))
                    found = True                
            # Check if it is a subgroup
            elif isinstance(item, Group):
                item.analyse()
                items.append(Nodes.Parenthesis(self.machine, value=None, child=item))
                found = True          
            # Else: don't know what to do with it...
            if not found:
                print self.machine.expr
                print 'Error: the term', item, 'is neither part of the allowed vocabulary or a parameter of your model.'
                exit(0)
        self.items = items
