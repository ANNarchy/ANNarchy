from Variable import Variable

import re
import pprint

class Master2:
    def __init__(self):
        self._variables = []
        
    def _transform_expr_in_variable(self, expr):

        var = {}
        var['type'] = 'local'
        
        try:
            constraints = {}
            equation, constraint = expr.split(':')
            # eval the constraints
            for con in constraint.split(','):
                try:
                    key, value = con.split('=')
                    key = re.sub('\s+', '', key)
                    constraints[key] = value
                    
                except ValueError:
                    if con.find('population') !=-1:
                        var['type'] = 'global'
                    else:
                        print 'Error: constraint statement', con, 'is wrong.'
                        continue
            
        except ValueError:
            equation = expr # there are no constraints
            
        finally:
                     
            #
            # evaluate the equation
            try:
                lside, rside = equation.split('=')
            
                name = re.findall("[\w\s]+\/[\w\s]+", lside)
                if name == []:
                    #found direct value assignment, either init value or equation
                    name = re.findall("[\w]+", lside)
                    var['name'] = name
                    try:
                        rvalue = float(rside) #just test conversion (if rside contains Uniform or something else we chose the except path)
                        var['var'] = Variable(init=rside, **constraints)
                    except ValueError:
                        var['var'] = Variable(eq=rside, **constraints)
                    
                else:
                    # found an ODE
                    name = re.findall("(?<=d)[\w\s]+(?=/)", lside)
                    var['name'] = name
                    var['var'] = Variable(eq = rside, **constraints)
                    
            except ValueError:
                if not 'init' in constraints.keys():
                    print 'WARNING: no default value for', equation
                
                var['name'] = equation
                var['var'] = Variable(**constraints)                    
            
        found = False   
        for iter in self._variables:
            if iter['name'] == var['name']:
                found = True
                iter['var'] + var['var']
            
        if not found:
            self._variables.append(var)
    
    def _convert(self, set_of_eq):

         # replace the ,,, by empty space and split the result up
        tmp_set = re.sub('\s+\.\.\.\s+', ' ', set_of_eq).split('\n')
        
        # check each line      
        for expr in tmp_set:
            expr = re.sub('\#[\s\S]+', ' ', expr) # remove comments
            expr = re.sub('\s+', ' ', expr) # remove additional tabs etc.
            if expr == ' ' or len(expr)==0:
                continue
            
            self._transform_expr_in_variable(expr)

        # check created variables        
        for var in self._variables:
            var['var']._validate()

class RateNeuron(Master2):
    def __init__(self, parameters, equations, extra_values=None, functions=None):
        Master2.__init__(self)
        
        print 'variables:'
        self._convert(parameters +'\n'+ equations) 
        pprint.pprint( self._variables, depth=4 )
        print '\n'
        
class SpikeNeuron(Master2):

    def __init__(self, parameters, equations, spike, reset, extra_values=None, functions=None ):
        Master2.__init__(self)
        
        print 'variables:'
        self._convert(parameters +'\n'+ equations) 
        pprint.pprint( self._variables, depth=4 )
        print '\n'
            
        print 'spike:\n', spike
        print 'reset:\n', reset
