"""

    Nodes.py
    
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
import Analyser
import Tree
            
## Object transforming the expression into an ordered tree
class TreeBuilder(object):

    def __init__(self, machine, expr_list):
        self.machine = machine
        self.expr_list = expr_list
    
    def build(self):
        # Split the list around the equal sign
        id_equal = self.expr_list.index(EQUAL)
        import Group
        self.left_group, self.right_group = Group.Group(self.machine, self.expr_list[:id_equal]), Group.Group(self.machine, self.expr_list[id_equal+1:])
        # Start the group analysis
        for group in [self.left_group, self.right_group]:
            group.group()
            group.analyse()       
        # Return ordered tree
        tree = Equal(self.machine, left=self.left_group, right=self.right_group).build_hierarchy()
        return tree
        
# Node class
class Node(object):
    """Base node of the tree processing a group of expressions.
    
    The hierarchize() method analyses its content and adds unary/binary/ternary operators if needed.
    """
    def __init__(self, machine):
        self.machine = machine
        
    def __repr__(self):
        return self.cpp()

    def hierarchize(self, side):
        """ Create a hierarchy using the recognized nodes. """
        if isinstance(side, Parenthesis):
            return side
        if isinstance(side, list) :
            if len(side)==1: # The group has only one item
                import Group
                if isinstance(side[0], Group.Group): # A term between brackets.
                    newside = self.hierarchize(side[0])
                    newside.parent= self
                    return newside
                else: # Terminal leaf
                    newside = side[0]
                    newside.parent= self
                    return newside
        for i in range(len(side)): # if then else
            if isinstance(side[i], If):
                newside = If(self.machine, value = 'if', child=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): # logical operator
            if isinstance(side[i], Logical):
                newside = Logical(self.machine, 
                                    value=side[i].value,
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): # not operator
            if isinstance(side[i], Not):
                newside = Not(self.machine,
                                value = side[i].value,
                                child = side[i+1:] )
                newside.parent= self
                return newside
        for i in range(len(side)): # comparator
            if isinstance(side[i], Comparator):
                newside = Comparator(self.machine, 
                                    value=side[i].value,
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): # add
            if isinstance(side[i], PlusOperator):
                newside = PlusOperator(self.machine, 
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): # sub
            if isinstance(side[i], MinusOperator):
                if i >0:
                    newside = MinusOperator(self.machine, 
                                        left=(side[:i]), 
                                        right=(side[i+1:]) )
                else:
                    newside = MinusOperator(self.machine, 
                                        left=[Constant(self.machine, value='')], 
                                        right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): #mult
            if isinstance(side[i], MultOperator):
                newside = MultOperator(self.machine, 
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)-1, 0, -1): #div
            if isinstance(side[i], DivOperator):
                newside = DivOperator(self.machine, 
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)):# power function
            if isinstance(side[i], PowOperator):
                newside = PowOperator(self.machine, 
                                    left=(side[:i]), 
                                    right=(side[i+1:]) )
                newside.parent= self
                return newside
        for i in range(len(side)): # No operators, only functions are left
            if isinstance(side[i], Function):
                newside = Function(self.machine,
                                value = side[i].value,
                                child = side[i+1] )
                newside.parent= self
                return newside
            elif isinstance(side[i], GlobalFunction):
                newside = GlobalFunction(self.machine,
                                value = side[i].value,
                                child = side[i].child )
                newside.parent= self
                return newside
        # Not found
        print self.machine.expr
        print 'Error: could not analyse the expression.'
        exit(0)
    
# Equal object        
class Equal(Node):
    """ Top node of the hierarchy. There should be only one Equal object."""
    def __init__(self, machine, left, right):
        Node.__init__(self, machine)
        self.machine = machine
        self.left = left
        self.right = right
        self.order=0
                
    def cpp(self):
        if self.order == 1:
            return self.left.cpp() + " += dt_*" + self.right.cpp()
        return self.left.cpp() + " = " + self.right.cpp()
                
    def latex(self):
        return self.left.latex() + " = " + self.right.latex()
        
    def build_hierarchy(self):
        left=None
        right=None
        # left member
        if len(self.left) == 1:
            if not isinstance(self.left[0], MainVariable) and not isinstance(self.left[0], Gradient):
                print self.machine.expr 
                print 'Error: the left term should be the updated variable'
                exit(0)
            else:
                left = self.left[0]
        else: # there is at least one operator
            left = self.hierarchize(self.left) 
        self.left = left
        self.left.parent= self  
                    
        # right member 
        right = self.hierarchize(self.right) 
        self.right = right
        self.right.parent= self
        return self

#############################
# Simple objects
#############################
        
# Leaf (no need to go further)
class Leaf(object):
    def __repr__(self):
        return self.cpp()       
        
# Constants
class Constant(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
        
    def cpp(self):
        if self.value in PI:
            return ' ' + cpp_equivalent(self.value) + ' '
        elif self.value in TRUE:
            return ' ' + cpp_equivalent(self.value) + ' '
        elif self.value in FALSE:
            return ' ' + cpp_equivalent(self.value) + ' '
        elif self.value == '': # case of negative numbers
            return ''
        return ' ' + str(float(self.value)) + ' '
        
    def latex(self):
        if self.value in PI:
            return '{\\pi}'
        return '{' + str(self.value) + '}'

##################################
# Variables and parameters
##################################

# Variable of the neuron/synapse, but not the one updated here        
class Variable(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
    def cpp(self):
        if isinstance(self.machine.analyser, Analyser.SynapseAnalyser):
            if str(self.value) in self.machine.analyser.local_variables_names:
                suffix = '[i] '
            else:
                suffix = ' '
        else:
            suffix = '[i] '
        if self.value == 't':
            return ' time_ '
        return ' ' + str(self.value)+'_'+suffix
    def latex(self):
        return '{\\text{'+str(self.value) + '}}'

# Main variable updated        
class MainVariable(Variable):
    def __init__(self, machine, value):
        Variable.__init__(self, machine, value)
    def cpp(self):
        if isinstance(self.machine.analyser, Analyser.SynapseAnalyser):
            if str(self.value) in self.machine.analyser.local_variables_names:
                suffix = '[i] '
            else:
                suffix = ' '
        else:
            suffix = '[i] '
        return ' ' + str(self.value)+'_'+suffix

# Parameter if the neuron/synapse        
class Parameter(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
    def cpp(self):
        return ' ' + str(self.value) + '_'
    def latex(self):
        return '{\\text{'+str(self.value) + '}}'

# Gradient on the main variable        
class Gradient(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
    def cpp(self):
        if isinstance(self.machine.analyser, Analyser.SynapseAnalyser):
            if str(self.value) in self.machine.analyser.local_variables_names:
                suffix = '[i] '
            else:
                suffix = ' '
        else:
            suffix = '[i] '
        return ' ' + str(self.value)+'_'+suffix
    def latex(self):
        return '{\\frac{d\\text{'+str(self.value) + '}}{dt}}'

# Weighted sum of inputs        
class PSP(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value[0]
    def cpp(self):
        if not self.value in self.machine.targets:
            print self.machine.expr
            print 'Warning: the target', self.value, 'does not exist on this neuron. The sum will be 0.0.'
            return ' 0.0 '
        return ' sum(i, ' + str(self.machine.targets.index(self.value))+') '
    def latex(self):
        return '{\\sum_{i}^{'+str(self.value)+'} w_{i} \\cdot \\text{rate}_{i}}'

# Presynaptic variable (e.g. pre.rate)        
class PreVariable(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
    def cpp(self):
        variable =  self.value.split('.')[1]
        if variable=='rate':
            return ' (*pre_rates_) [ rank_ [i] ] '
        return ' pre_population_->getSingle'+variable.capitalize()+'( rank_[i] ) '
    def latex(self):
        return '{\\text{'+str(self.value)+'}}'
      
# Postsynaptic variable (e.g. post.rate)   
class PostVariable(Leaf):
    def __init__(self, machine, value):
        self.machine = machine
        self.value = value
    def cpp(self):
        variable =  self.value.split('.')[1]
        if variable=='rate':
            return ' (*post_rates_) [ post_neuron_rank_ ] '
        return ' post_population_->getSingle'+variable.capitalize()+'( post_neuron_rank_ ) '
    def latex(self):
        return '{\\text{'+str(self.value)+'}}'
        
########################
# Unitary operators
########################
        
        
# Parenthesized term   
class Parenthesis(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        Node.__init__(self, machine)
        self.machine = machine
        self.value = value
        self.child = child
        if self.child != None and hierarchize:
            self.child = self.hierarchize(self.child)
            self.child.parent=self
            
    def cpp(self):
        if self.child != None:
            return '(' + self.child.cpp() + ')'  
        return '()'
        
    def latex(self):
        return '('+self.child.latex()+')'
        
# Mathematical functions    
class Function(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        Node.__init__(self, machine)
        self.machine = machine
        self.value = value
        self.child = child
        if self.child != None and hierarchize:
            self.child = self.hierarchize(self.child)
            self.child.parent=self
            
    def cpp(self):
        if self.child != None:
            return str(cpp_equivalent(self.value))+self.child.cpp()
        return str(self.value)+'()'
        
    def latex(self):
        if self.child != None:
            if self.value in POS:
                return '{('+self.child.latex()+')^+}' 
            elif self.value in NEG:
                return '{('+self.child.latex()+')^-}' 
            elif self.value in SQRT:
                return '{' + latex_equivalent(self.value)+'{'+self.child.latex()+'}}' 
            else:
                return '{' + latex_equivalent(self.value)+'{('+self.child.latex()+')}}'  
        return '{\\text{'+str(self.value)+'}()}'
        
# Not operator    
class Not(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        Node.__init__(self, machine)
        self.machine = machine
        self.value = value
        self.child = child
        if self.child != None and hierarchize:
            self.child = self.hierarchize(self.child)
            self.child.parent=self
            
    def cpp(self):
        if self.child != None:
            return str(cpp_equivalent(self.value))+'('+self.child.cpp()+')'  
        return str(self.value)+'()'
        
    def latex(self):
        return '{\\bar{'+str(self.value)+'}}'
        
        
# Global function (mean, max, min)    
class GlobalFunction(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        Node.__init__(self, machine)
        self.machine = machine
        self.value = value
        self.child = child
        if self.child != None and hierarchize:
            if len(self.child) != 1:
                print 'Error: only one term should be specified inside the', self.value, 'function.'
                exit(0)
            self.child=self.child[0]
            self.pop = 'pre' if self.child.find('pre.') != -1 else 'post'
            if self.child.find('.') == -1:
                self.variable = self.child
            else:
                self.variable = self.child.split('.')[1]
            # Tell the parser that it should generate the corresponding functions
            self.machine.analyser.global_operations[self.pop].append({'variable': self.variable, 'function': self.value})
                
    def cpp(self):
        if self.child != None:
            # TODO: save which variables are called. neuron?
            if isinstance(self.machine.analyser, Analyser.NeuronAnalyser):
                return ' get' + self.value.capitalize() + self.variable.capitalize() + '()'
            else:
                return ' ' + self.pop + '_population_->get' + self.value.capitalize() + self.variable.capitalize() + '() '
            
        return str(self.value)+'()'
        
    def latex(self):
        return '{\\text{'+str(self.value)+'}{'+ str(self.child)+ '}}'
        
######################
# Ternary operator    
######################

# IF node    
class If(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        self.machine = machine
        self.value = value
        self.child = child
        if self.child is not None and hierarchize:
            id_then = -1
            id_else = -1
            nb_if = 0
            for rk, item in enumerate(self.child):
                if isinstance(item, If):
                    nb_if += 1
                elif isinstance(item, Then):
                    if nb_if == 0:
                        id_then =  rk
                elif isinstance(item, Else):                    
                    if nb_if == 0:
                        id_else =  rk
                    else:
                        nb_if -= 1
                    
            if id_then is -1 or id_else is -1:
                print 'Error in analysing', self.machine.expr
                print 'The conditional should use the (if A then B else C) structure.'
                exit(0)
                
            import Group
            self.cond = self.hierarchize(Group.Group(self.machine, self.child[:id_then]))
            self.then = self.hierarchize(Group.Group(self.machine, self.child[id_then+1: id_else]))
            self.elsestatement = self.hierarchize(Group.Group(self.machine, self.child[id_else+1:]))
            
    def cpp(self):
        if self.child:
            return '( (' +  self.cond.cpp() + ') ? ' + self.then.cpp() + ' : ' + self.elsestatement.cpp() + ')'
        else: 
            return ' if ' 
    def latex(self):
        return ''      
        
# Then node        
class Then(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        self.machine = machine
        self.value = value
        self.child = child
            
    def cpp(self):
        return ' then ' 
    def latex(self):
        return ''   

# Else node           
class Else(Node):
    def __init__(self, machine, value, child=None, hierarchize=True):
        self.machine = machine
        self.value = value
        self.child = child
            
    def cpp(self):
        return ' else ' 
    def latex(self):
        return ''          

###########################
# Binary operators
###########################

# Basic Operator
class Operator(Node):
    def __init__(self, machine, value, left, right, hierarchize=True):
        Node.__init__(self, machine)
        self.machine = machine
        self.value = value
        self.left= left
        self.right=right
        if self.left != None and  hierarchize:
            self.left = self.hierarchize(self.left)
            self.left.parent = self
        if self.right != None and hierarchize:
            self.right = self.hierarchize(self.right)
            self.right.parent = self
    
    def cpp(self):
        if self.left != None and self.right != None:
            if str(self.value) == "+" or str(self.value) == "-" :
                return self.left.cpp() + str(self.value) + self.right.cpp() 
            else:
                return ' (' + self.left.cpp() + ')' + str(self.value) + '(' + self.right.cpp() + ') '
        return ' (' + str(self.value) + ') '
 
# Comparators (< > <= >=)       
class Comparator(Operator):
    def __init__(self, machine, value='<', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
        
    def cpp(self): # comparators are already in C++
        if self.left is not None:
            return self.left.cpp() + cpp_equivalent(self.value) + self.right.cpp() 
        return str(self.value)
            
    def latex(self):
        if self.value is '>':
            return '{>}'
        elif self.value is '>=':
            return '{\geq}'
        elif self.value is '<':
            return '{<}'
        elif self.value is '<=':
            return '{\leq}'
        elif self.value is '!=':
            return '{\neq}'
        return '{' + str(self.value) + '}'
        
# Logical operator (and, or)
class Logical(Operator):
    def __init__(self, machine, value='and', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
        
    def cpp(self): # comparators are already in C++
        if self.left != None:
            return ' (' + self.left.cpp() + ') ' + str(cpp_equivalent(self.value)) + ' (' + self.right.cpp() + ') '
        return str(self.value)
            
    def latex(self):
        return "\\text{and}"

# +    
class PlusOperator(Operator):   
    def __init__(self, machine, value='+', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
        
    def latex(self):
        if self.left != None and self.right != None:
            if not isinstance(self.parent, MultOperator):
                return ' {' + self.left.latex() + str(self.value) + self.right.latex() + '} '
            else:
                return ' {(' + self.left.latex() + str(self.value) + self.right.latex() + ')} '
        return ''
# -    
class MinusOperator(Operator):   
    def __init__(self, machine, value='-', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
    def latex(self):
        if self.left != None and self.right != None:
            if not isinstance(self.parent, MultOperator):
                return ' {' + self.left.latex() + str(self.value) + self.right.latex() + '} '
            else:
                return ' {(' + self.left.latex() + str(self.value) + self.right.latex() + ')} '
        return ''

# *    
class MultOperator(Operator):   
    def __init__(self, machine, value='*', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
    def latex(self):
        if self.left != None and self.right != None:
            return ' {' + self.left.latex() + ' \cdot ' + self.right.latex() + '} '
        return ''

# /        
class DivOperator(Operator):   
    def __init__(self, machine, value='/', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)
    def latex(self):
        if self.left != None and self.right != None:
            return ' {\\frac{' + self.left.latex() + '}{' + self.right.latex() + '}} '
        return ''
# ^        
class PowOperator(Operator):   
    def __init__(self, machine, value='^', left=None, right=None, hierarchize=True):
        Operator.__init__(self, machine, value, left, right, hierarchize)

    def cpp(self):
        return ' pow('+str(self.left) + ' , ' + str(self.right)+') '

    def latex(self):
        return ' {('+self.left.latex() + ')}^{' + self.right.latex()+'} '



