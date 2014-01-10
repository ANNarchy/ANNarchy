from Variable import Variable

import re

class Master2:
    def __init__(self):
        pass

    def _transform_expr_in_variable(self, expr):
        
        var = {}
        print 'process:\n\t', expr    
    
        try:
            constraints = {}
            equation, constraint = expr.split(':')
            # eval the constraints
            for con in constraint.split(','):
                try:
                    key, value = con.split('=')
                    constraints[key] = value
                    
                except ValueError:
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
                    name = re.findall("[\w]+", lside)
                    var['name'] = name
                    var['type'] = 'parameter'
                    var['var'] = Variable(init=rside, eq = None)
                else:
                    name = re.findall("(?<=d)[\w\s]+(?=/)", lside)
                    var['name'] = name
                    var['type'] = 'variable'
                    var['var'] = Variable(init=0.0, eq = rside)
                    
            except ValueError:
                name = equation
                var['name'] = name
                var['init'] = 0.0
                var['type'] = 'parameter' 
                    
        print 'detected:\n\t', name,'with eq:', equation,'and constraints:', constraints            
               
        return var
    
    def _convert(self, set_of_eq):
        ret = []
               
         # replace the ,,, by empty space and split the result up
        tmp_set = re.sub('\s+\.\.\.\s+', ' ', set_of_eq).split('\n')
        
        # check each line      
        for expr in tmp_set:
            expr = re.sub('\#[\s\S]+', ' ', expr) # remove comments
            expr = re.sub('\s+', ' ', expr) # remove additional tabs etc.
            if expr == ' ' or len(expr)==0:
                continue
            
            ret.append(self._transform_expr_in_variable(expr))
        
        return ret

class RateNeuron(Master2):
    def __init__(self, parameters, variables, functions=None):
        Master2.__init__(self)
        
        self._parameters = self._convert(parameters) 
        print 'parameters:\n', self._parameters 

        self._variables = self._convert(variables) 
        print 'variables:\n', self._variables
        
class SpikeNeuron(Master2):

    def __init__(self, parameters, variables, spike, reset, functions=None ):
        Master2.__init__(self)
        
        self._parameters = self._convert(parameters) 
        print 'parameters:\n', self._parameters 

        self._variables = self._convert(variables) 
        print 'variables:\n', self._variables
            
        print 'spike:\n', spike
        print 'reset:\n', reset
