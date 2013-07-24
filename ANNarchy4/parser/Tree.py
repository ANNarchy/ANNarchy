from Definitions import *
from Nodes import *
import copy

# State machine for a single expression
class Tree:

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
        
        # Perform the analysis
        self.analyse()
        
        
    def analyse(self):
        # Expand all whitespaces
        self.expr_list = self.expand_whitespaces(self.expr).split(' ')
        while '' in self.expr_list:
            self.expr_list.remove('')
        # Check for an equal sign
        if not EQUAL in self.expr_list:
            print 'Error in the evaluation of', self.name, '(', self.expr, '): there is no equal sign.'
            exit(0)
        # Start the recursive analysis
        self.tree = TreeBuilder(self, self.expr_list).build()
        # Move the tree so that only the target variabe is on the left.
        # Search for the gradient on the left side if any
        if lookup(self.tree.left, Gradient) != None:
            self.simplifiedtree = simplify(copy.copy(self.tree), Gradient)
            self.simplifiedtree.order = 1
        elif lookup(self.tree.left, MainVariable) != None:
            self.simplifiedtree = simplify(copy.copy(self.tree), MainVariable)
            self.simplifiedtree.order = 0
        
    def expand_whitespaces(self, expr):
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
    if isinstance(tree.left, PlusOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = MinusOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = MinusOperator(tree.machine, left=tree.right, right=tree.left.left, hierarchize=False)
            tree.left = tree.left.right
    elif isinstance(tree.left, MinusOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = PlusOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = MinusOperator(tree.machine, left=tree.left.left, right=tree.right, hierarchize=False)
            tree.left = tree.left.right
            
    elif isinstance(tree.left, MultOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = DivOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = DivOperator(tree.machine, left=tree.right, right=tree.left.left, hierarchize=False)
            tree.left = tree.left.right
        
    elif isinstance(tree.left, DivOperator):
        if tree.left.left.has_target: # the left branch has the target
            tree.right = MultOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
            tree.left = tree.left.left
        else:
            tree.right = DivOperator(tree.machine, left=tree.left.left, right=tree.right, hierarchize=False)
            tree.left = tree.left.right
        
    elif isinstance(tree.left, PowOperator):
        if tree.left.left.has_target: # the left branch has the target
            if not float(tree.left.right.value) == 2.0:
                tree.right = PowOperator(tree.machine, left=tree.right, right=tree.left.right, hierarchize=False)
                tree.right.right.value = str(1.0/float(tree.right.right.value))
                tree.left = tree.left.left
            else:
                tree.right = Function(tree.machine, value = 'sqrt', child=tree.right, hierarchize=False)
                tree.left = tree.left.left                
        else:
            print 'Error: can not solve a^x = b yet.'
            exit(0)
            
    else:
        print 'Error: one node is not invertible in the left member.'
        exit(0)

    return simplify(tree, target)

