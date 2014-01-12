"""

    Population.py
    
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
from Variable import Variable

import Random

import re


class Master2(object):
    """
    Root object for all neuron or synapse base class implementations.
    
    The transformation code of several data used are similar. Special cases are handled 
    through the several derived classes.
    """
    def __init__(self):
        """
        Constructor.
        """
        self._variables = {}
        self._order = []
        
    def _transform_expr_in_variable(self, expr, is_eq=False):
        """
        Private function.
        
        This function transforms a single expression into a ANNarchy Variable representation and 
        should be called only by the _convert function.
        
        Parameter:
        
        * *is_eq*: determines if the expression belongs to equation part or parameter part.
        
        Return:
        
        * instance of ANNarchy.core.Variable
        """
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
                    con = con.replace(' ','')
                    if con.find('population') !=-1:
                        var['type'] = 'global'
                    elif con.find('postsynaptic') !=-1:
                        var['type'] = 'global'
                    else:
                        print expr
                        print "WARNING: constraint statement '", con, "' is wrong or no arguments provided."
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
                    try:
                        #just test conversion (if rside contains Uniform or something else we chose the except path)
                        rvalue = float(rside) 
                        if is_eq:
                            var['var'] = Variable(eq=equation, **constraints)
                        else:
                            var['var'] = Variable(init=rside, **constraints)
                    except ValueError:
                        rand_dist = re.findall("[\w\s]+(?=\()", rside) #matches Uniform(), Normal() but also sum()
                        found = False
                        for rand in rand_dist:
                            rand = rand.replace(' ','')
                            
                            if rand in Random.RandomDistribution().keywords():
                                found = True
                                args = re.findall("[\d\.]+", equation)
                                rand_obj = getattr(Random, rand)(float(args[0]),float(args[1]))
                                var['var'] = Variable(eq=rand_obj, **constraints)
                                
                        if not found:
                            var['var'] = Variable(eq=equation, **constraints)
                    
                else:
                    # found an ODE
                    name = re.findall("(?<=d)[\w\s]+(?=/)", lside)
                    var['var'] = Variable(eq = equation, **constraints)
                    
            except ValueError:
                if not 'init' in constraints.keys():
                    print 'WARNING: no default value for', equation
                
                name = equation
                var['var'] = Variable(**constraints)                    
            
        # during other operations a list of single characters is created
        # so, now we convert list of single characters to a string
        name = ''.join(name) 
        
        if is_eq:
            self._order.append(name)

        return name, var
    
    def _prepare_string(self, stream):
        """
        The function splits up the several equations, remove comments and unneeded spaces or tabs.
        
        Parameter:
        
        *  *stream*: string contain a variable, parameter or similar set, e.g.
        
        .. code-block:: python 

            \"\"\" 
                tau = 10.0 : population
                baseline = 0.0 
            \"\"\"
        
        Return:
            
        an array contain the several strings, according the above example:
        
        .. code-block:: python
            
            [ ' tau = 10.0 : population ', ' baseline = 0.0 ' ] 
        """
        expr_set = []
        
        # replace the ,,, by empty space and split the result up
        tmp_set = re.sub('\s+\.\.\.\s+', ' ', stream).split('\n')
        
        for expr in tmp_set:
            expr = re.sub('\#[\s\S]+', ' ', expr)   # remove comments
            expr = re.sub('\s+', ' ', expr)     # remove additional tabs etc.
                    
            if expr == ' ' or len(expr)==0: # through beginning line breaks or something similar empty strings are contained in the set
                continue
            
            expr_set.append(expr)
        
        return expr_set
        
    def _convert(self, parameters, equations):
        """
        Private function.
        
        This function transforms the different possible equation types into an easier
        parsable data representation. This function is called from the different Neuron/Synapse
        constructors. 
        """
        tmp_par = self._prepare_string(parameters)
        tmp_eq = self._prepare_string(equations)
                
        # check each line      
        for expr in tmp_par:
            name, var = self._transform_expr_in_variable(expr)
            self._variables[name] = var
            
        # check each line      
        for expr in tmp_eq:
            name, var = self._transform_expr_in_variable(expr, True)

            if name in self._variables.keys():
                self._variables[name]['var'] + var['var'] 
            else:
                self._variables[name] = var

        # check created variables        
        for var in self._variables.keys():
            self._variables[var]['var']._validate()
            
    @property
    def variables(self):
        return self._variables
    
    @property
    def order(self):
        return self._order
    
    