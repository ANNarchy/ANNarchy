from Definitions import *
import Tree
import re

from ANNarchy4.core.Random import RandomDistribution

# Main analyser class for neurons
class NeuronAnalyser:
    
    def __init__(self, neuron, targets):
    
        self.neuron = neuron 
        self.targets = targets
        self.analysed_neuron = []
        self.parameters_names = []
        self.variables_names = []
        self.trees = []
        
    def parse(self):
    
        # Determine parameters and variables
        for value in self.neuron:
            if not 'name' in value.keys():
                print 'Error: dictionary must have a name attribute.'
                exit(0)

            if 'var' in value.keys():
                self.variables_names.append(value['name'])
            else: # A parameter
                self.parameters_names.append(value['name'])
                     
        # Perform the analysis
        for value in self.neuron:
            if 'var' in value.keys(): # A variable which needs to be analysed
                if value['var'].init != None:
                    init_value = value['var'].init
                else:
                    init_value = 0.0

                if value['var'].eq != None:
                    tree = Tree.Tree(self, value['name'], value['var'].eq)
                    self.trees.append(tree)
                    self.analysed_neuron.append( 
                        {'name': value['name'],
                            'type': 'variable',
                            'init': self.init_variable(value['name'], init_value),
                            'def': self.def_variable(value['name']),
                            'cpp' : tree.cpp() +';'} )
                else:
                    self.analysed_neuron.append( 
                        {'name': value['name'],
                            'type': 'variable',
                            'init': self.init_variable(value['name'], init_value),
                            'def': self.def_variable(value['name']),
                            'cpp' : ''} )

            else: # A parameter
                if 'init' in value.keys():
                    init_value = value['init']
                else:
                    init_value = 0.0
                self.analysed_neuron.append( 
                    {'name': value['name'],
                     'type': 'parameter',
                     'init': self.init_parameter(value['name'], init_value),
                     'def': self.def_parameter(value['name']),
                     'cpp' : '' } )
                     
        return self.analysed_neuron
                
    def def_parameter(self, name):
        return DATA_TYPE +' '+ name+';'

    def def_variable(self, name):
        return 'std::vector<'+DATA_TYPE+'> '+ name+';'

    def init_parameter(self, name, value):
        if isinstance(value, RandomDistribution):
            return name + ' = ('+ value.genCPP() +').getValue();'
        else:
            return name + ' = ' + str(value) + ';'
        
    def init_variable(self, name, value):
        if isinstance(value, RandomDistribution):
            return name+' = ('+ value.genCPP() +').getValues(nbNeurons_);' # after this call the instantiation object still remains in memory -_-
        else:
            return name+' = std::vector<' + DATA_TYPE + '> ' +  '(nbNeurons_, '+str(value)+');'
        
    def latex(self):
        code =""
        for tree in self.trees:
            code += '    ' + tree.latex() + '\n\n'
        return code

# Main analyser class for learning rules
class LearningRuleAnalyser:
    
    def __init__(self, learning_rule):
    
        self.learning_rule = learning_rule 
        self.analysed_neuron = []
        self.parameters_names = []
        self.variables_names = []
        self.targets=None
        self.trees = []
        
    def parse(self):
    
        # Determine parameters and variables
        for value in self.learning_rule:
            if not 'name' in value.keys():
                print 'Error: dictionary must have a name attribute.'
                exit(0)
            if 'eq' in value.keys(): # A variable which needs to be analysed
                self.variables_names.append(value['name'])
            else: # A parameter
                self.parameters_names.append(value['name'])
                
        # Identify the local variables (synapse-specific) from the global ones (neuron-specific)
        dependencies={}
        for value in self.learning_rule:
            if value['name'] in self.variables_names: # only variables count
                dep = []
                if not value['eq'].find('pre.') == -1: # directly depends on pre
                    dep.append('pre')
                else:
                    for ovar in self.variables_names: # check indirect dependencies
                        if ovar != value['name']: # self-dependencies do not count
                            code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])'+ovar+'(?P<post>[^\_a-zA-Z0-9])', value['eq'])
                            if len(code) > 0: # wont work
                                dep.append(ovar)
                    
                        
                dependencies[value['name']] = dep

        self.local_variables_names, self.global_variables_names = self.sort_dependencies(dependencies)
        
                     
        # Perform the analysis
        for value in self.learning_rule:
            if value['name'] in self.local_variables_names: # A variable which needs to be analysed
                if 'init' in value.keys():
                    init_value = value['init']
                else:
                    init_value = 0.0
                tree = Tree.Tree(self, value['name'], value['eq'])
                self.trees.append(tree)
                self.analysed_neuron.append( 
                    {'name': value['name'],
                     'type': 'local',
                     'init': self.init_local_variable(value['name'], init_value),
                     'cpp' : tree.cpp() +';'} )
            elif value['name'] in self.global_variables_names:
                if 'init' in value.keys():
                    init_value = value['init']
                else:
                    init_value = 0.0
                tree = Tree.Tree(self, value['name'], value['eq'])
                self.trees.append(tree)
                self.analysed_neuron.append( 
                    {'name': value['name'],
                     'type': 'global',
                     'init': self.init_global_variable(value['name'], init_value),
                     'cpp' : tree.cpp() +';'} )
            else: # A parameter
                if 'init' in value.keys():
                    init_value = value['init']
                else:
                    init_value = 0.0

                self.analysed_neuron.append( 
                    {'name': value['name'],
                     'type': 'parameter',
                     'init': self.init_parameter(value['name'], init_value),
                     'cpp' : '' } )
                     
        return self.analysed_neuron
        
        
        
    def init_parameter(self, name, value):
        return DATA_TYPE + ' ' + name + ' = ' + str(value) + ';';
        
    def init_local_variable(self, name, value):
        return 'vector<' + DATA_TYPE + '> ' + name + '(nbSynapses_, '+str(value)+');'
        
    def init_global_variable(self, name, value):
        return 'vector<' + DATA_TYPE + '> ' + name + '(pre_->nbNeurons_(), '+str(value)+');'
                
    def latex(self):
        code =""
        for tree in self.trees:
            code += '    ' + tree.latex() + '\n\n'
        return code
        
    def sort_dependencies(self, dependencies):
        """ Inspects the dependencies between all variables to decide whether they are local or global."""
    
        sorted_dependencies = {} 
        for name, deps in dependencies.items():
            sorted_dependencies[name] = False
        for name, deps in dependencies.items():
            if 'pre' in deps:
                sorted_dependencies[name] = True
        stable = False
        while not stable:
            stable = True
            for name, deps in dependencies.items():
                is_dep_pre = False
                for dep in deps:
                    if not dep == 'pre':
                        is_dep_pre = is_dep_pre or sorted_dependencies[dep]
                        if not is_dep_pre == sorted_dependencies[name]:
                            stable = False
                            sorted_dependencies[name] = is_dep_pre
        
        localvar = []
        globalvar = []
        for name in dependencies.keys():
            if sorted_dependencies[name]:
                localvar.append(name)
            else:
                globalvar.append(name)
        return localvar, globalvar
                

