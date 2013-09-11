"""
Population.py
"""
import re
import numpy as np

# ANNarchy core informations
import ANNarchy4.core.Global as Global
from ANNarchy4.core.Random import RandomDistribution

from ANNarchy4.core.Variable import Variable
from ANNarchy4 import parser
import copy

class Population(object):
    """
    Population generator base class
    """
    def __init__(self, population):
        self.class_name = 'Population'+str(population.id)
        
        self.header = Global.annarchy_dir+'/build/'+self.class_name+'.h'
        self.body = Global.annarchy_dir+'/build/'+self.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/pyx/'+self.class_name+'.pyx'
        
        self.population = population
        
        self.rand_objects = []
        self.targets = []
        self.neuron_variables = copy.deepcopy(self.population.neuron_type.variables)
        self.global_operations = { 'pre':[], 'post':[] }
        self.post_compilation_init = {}
        
    def _init_variables(self):
        """ Called after creation of the C++ objects to initialize variables with arrays."""
        self.population.set(self.post_compilation_init)

    def add_target(self, target):
        """
        Internal Function
        """
        self.targets.append(target)
        self.targets = list(set(self.targets))   # little trick to remove doubled entries
             
    def _get_value(self, name):
        """ Returns init value """
        if name in self.post_compilation_init.keys():
            return self.post_compilation_init[name]
        for var in self.neuron_variables:
            if var['name'] == name:
                if 'var' in var.keys(): # variable
                    return var['var'].init
                elif 'init' in var.keys(): # parameter
                    return var['init']
                else: # default
                    return 0.0 
        return None
        
    def _variable_names(self):
        """ Return names of all attached variables. """
        return [var['name'] for var in self.neuron_variables ]

    def _add_global_oparation(self, global_op):
        """
        Add the global operation to the populations dictionary. Besides this 
        function ensures the unique occurance of a variable/function pair.
        """

        if self.global_operations['post'] == []:
            self.global_operations['post'].append(global_op)
            
        for g_op2 in self.global_operations['post']:
            if global_op['variable'] == g_op2['variable'] and global_op['function'] == g_op2['function']:
                return
                
        self.global_operations['post'].append(global_op)
           
    def _add_value(self, name, value):
        """ 
        Add variable *name* to population. 
        
        **Attention**: this function print out an error, as it's not allowed 
        to add new variables / parameters to population object.
        """
        print "Error: it's not allowed to add new variables / parameters to population object."
        print 'Population:', self.population.name, 'variable', name, 'value', value

    def _update_value(self, name, value):
        """ 
        Update variable *name* with value *value*. 
        """
        values = next(( item for item in self.neuron_variables if item['name']==name ), None)
        if values:
            if 'var' in values.keys():
                if isinstance(value, (int, float)):       
                    values['var'].init = float(value)
                elif isinstance(value, Variable):
                    values['var'] = value
                elif isinstance(value, RandomDistribution):
                    values['var'].init = value
                elif isinstance(value, list):
                    if len(value) == self.population.size:
                        self.post_compilation_init[name] = value
                    else:
                        print 'Error: the variable', name, 'of population', self.population.name, 'must be initialized with a list of the same size', self.population.size                    
                elif isinstance(value, np.ndarray): # will be assigned after the constrution of the c++ objects
                    if value.shape == self.population.geometry or value.shape == (self.population.size, ):
                        self.post_compilation_init[name] = value
                    else:
                        print 'Error: the variable', name, 'of population', self.population.name, 'must be initialized with an array of the same shape', self.population.geometry  
                else:
                    print "Error: can't assign ", value , "(", type(value), ") to the variable "+name
            else:
                values['init'] = float(value)
        else:
            print "Error: variable / parameter "+name+" does not exist in population object."
            
    def _update_neuron_variables(self):
        """
        Updates the neuron variable dictionary e.g. :
            
            * global operations
            * replace RandomDistribution objects
            
        Should be called before code generation starts.
        """

        #   replace all RandomDistribution by rand variables with continous
        #   numbers and stores the corresponding call as local variable
        i = 0
        for value in self.neuron_variables:
            if 'var' not in value.keys(): 
                continue 

            if value['var'].eq == None: 
                continue 

            matches = re.findall('RandomDistribution\(.*?\)', value['var'].eq)
            for phrase in matches:
                part = phrase.split('RandomDistribution')[1]
                self.neuron_variables.append( 
                    { 'name': '_rand_'+str(i), 
                      'var': Variable(init = part) 
                    }
                )
                
                value['var'].eq = value['var'].eq.replace(phrase, '_rand_'+str(i))
                self.rand_objects.append(part)
                i += 1

        #   parse neuron
        self.neuron_parser = parser.NeuronAnalyser(
            self.neuron_variables, 
            self.targets
        )
        self.parsed_neuron, global_operations = self.neuron_parser.parse()
        
        #   attach needed global operations e.g. min, max, mean
        for g_op in global_operations['post']:
            self._add_global_oparation(g_op)        