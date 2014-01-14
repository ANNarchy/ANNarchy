"""

    Analyser.py

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
import Tree
import re

from ANNarchy4.core import Global
from ANNarchy4.core.Random import RandomDistribution
from ANNarchy4.core.SpikeVariable import SpikeVariable

def get_value_and_type(name, value):
    
    if 'var' in value.keys():
         # variables
        if value['var'].init != None:
            if isinstance(value['var'].init, RandomDistribution):
                 init_value = 0.0
            else:
                 init_value = value['var'].init
        else:
            init_value = 0.0
        
        if value['var'].type != None:
            cpp_type = value['var'].type
        else:
            cpp_type = type(init_value)
            
        if value['var'].type != type(init_value) and value['var'].type != None:
            if not Global.config['suppress_warnings']:
                print "'WARNING: type mismatch between provided type and initialization value of '", name,"' ('", value['var'].type,",", type(init_value),")."
             
    else:
        # parameter, have always an initial value,
        # but no type information
        init_value = value['init']
        cpp_type = type(init_value)
         
    return cpp_type, init_value

# Main analyser class for neurons
class NeuronAnalyser(object):

    def __init__(self, neuron, targets, pop_name):

        self.neuron = neuron
        self.pop_name = pop_name
        self.targets = targets
        self.analysed_neuron = {}
        self.parameters_names = []
        self.variables_names = []
        self.trees = []
        self.global_operations = {'pre': [], 'post': []}

    def parse(self):
        # Determine parameters and variables
        for name, value in self.neuron.iteritems():
            if value['type'] == 'local':
                self.variables_names.append(name)
            else: # A parameter
                self.parameters_names.append(name)

        # Perform the analysis
        for name, value in self.neuron.iteritems():
            
            if name in self.variables_names:
                cpp_type, init_value = get_value_and_type(name, value)

                #
                # basic stuff
                neur = {}
                neur['type'] = 'local'
                neur['cpp_type'] = cpp_type
                neur['def'] = self.def_variable(name)
                
                #
                # eq stuff
                if value['var'].eq != None:
                    
                    if isinstance(value['var'].eq, RandomDistribution):
                        neur['type'] = 'rand_variable'
                        neur['eq'] = value['var'].eq
                        self.analysed_neuron[name] = neur
                        continue
                    
                    neur['eq'] = value['var'].eq
                    tree = Tree.Tree(self, name, value['var'].eq, self.pop_name)
                    if not tree.success: # Error while processing the equation
                        return None, None
                    self.trees.append(tree)

                    neur['init'] = self.init_variable(name, init_value)
                    neur['cpp'] = tree.cpp() + ';'
                else:
                    neur['init'] = self.init_variable(name, init_value)
                    neur['cpp'] = ''

                #
                # min, max
                if value['var'].min != None:
                    neur['min'] = value['var'].min

                if value['var'].max != None:
                    neur['max'] = value['var'].max

                if isinstance(value['var'],SpikeVariable):
                    neur['threshold'] = value['var'].threshold
                    neur['reset'] = value['var'].reset

                self.analysed_neuron[name] = neur

            else: # A parameter
                cpp_type, init_value = get_value_and_type(name, value)
                    
                self.analysed_neuron[name] =  {
                    'type': 'global',
                    'init': self.init_parameter(name, init_value),
                    'def': self.def_parameter(name),
                    'cpp' : '',
                    'cpp_type': cpp_type 
                }

#        for cpp in self.analysed_neuron:
#            print cpp['cpp']

        # Process the global operations
        self.global_operations = sort_global_operations(self.global_operations)

        return self.analysed_neuron, self.global_operations

    def def_parameter(self, name):
        return DATA_TYPE +' '+ name+'_;'

    def def_variable(self, name):
        return 'std::vector<'+DATA_TYPE+'> '+ name+'_;'

    def init_parameter(self, name, value):
        if isinstance(value, RandomDistribution):
            return name + '_ = ('+ value.genCPP() +').getValue();'
        else:
            return name + '_ = ' + str(value) + ';'

    def init_variable(self, name, value):
        if isinstance(value, RandomDistribution):
            return name+'_ = ('+ value.genCPP() +').getValues(nbNeurons_);' # after this call the instantiation object still remains in memory -_-
        else:
            return name+'_ = std::vector<' + DATA_TYPE + '> ' +  '(nbNeurons_, '+str(value)+');'

    def latex(self):
        code =""
        for tree in self.trees:
            code += '    ' + tree.latex() + '\n\n'
        return code

# Main analyser class for synapses
class SynapseAnalyser(object):

    def __init__(self, synapse, targets_pre=[], targets_post=[]):

        self.synapse = synapse
        self.analysed_synapse = []
        self.parameters_names = []
        self.variables_names = []
        self.targets_pre=targets_pre # Need the list of targets for each population to allow pre.sum(exc) or post.sum(dopa)
        self.targets_post=targets_post
        self.targets = list(set(self.targets_pre + self.targets_post))
        self.targetIDs=None # What is it doing?
        self.trees = []
        self.global_operations = {'pre': [], 'post': []}

    def parse(self):

        # Determine parameters and variables
        for name, value in self.synapse.iteritems():
            if value['type'] == 'local':
                self.variables_names.append(name)
            elif value['type'] == 'global':
                self.parameters_names.append(name)
            else:
                continue
                
        # Identify the local variables (synapse-specific) from the global ones (neuron-specific)
        dependencies={}
        for name, value in self.synapse.iteritems():
            if name in self.variables_names: # only variables count
                dep = []
                if value['var'].eq == None:
                    continue
                else:
                    value['var'].eq += ' ' # in case a variable to be extracted is at the end...
                if not value['var'].eq.find('pre.') == -1: # directly depends on pre
                    dep.append('pre')
                elif not value['var'].eq.find('value') == -1: # depends on value, but maybe as part of the name
                    code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])value(?P<post>[^\_a-zA-Z0-9])', value['var'].eq)
                    if len(code) > 0:
                        dep.append('value')
                else:
                    for ovar in self.variables_names: # check indirect dependencies
                        if ovar != name: # self-dependencies do not count
                            code = re.findall('(?P<pre>[^\_a-zA-Z0-9.])'+ovar+'(?P<post>[^\_a-zA-Z0-9])', value['var'].eq)
                            if len(code) > 0: # wont work
                                dep.append(ovar)
                dependencies[name] = dep

        #self.local_variables_names, self.global_variables_names = self.sort_dependencies(dependencies)
        self.local_variables_names = self.variables_names
        self.global_variables_names = self.parameters_names
        
        # Perform the analysis
        for name, value in self.synapse.iteritems():

            if value['type'] == 'local' and value['var'].eq != None:
                synapse = { }
                cpp_type, init_value = get_value_and_type(name, value)
                    
                tree = Tree.Tree(self, name, value['var'].eq)
                if not tree.success: # Error while processing the equation
                    return None, None
                self.trees.append(tree)

                # base data: name, type, init, cpp, cpp_type
                synapse['type'] = 'local'
                synapse['init'] = self.init_local_variable(name, init_value)                                
                
                synapse['name'] = name
                synapse['cpp'] = tree.cpp() +';'
                synapse['cpp_type'] = cpp_type
                synapse['eq'] = value['var'].eq
                
                #
                # extend by optional parameters
                if value['var'].min != None:
                    synapse['min'] = value['var'].min

                if value['var'].max != None:
                    synapse['max'] = value['var'].max

                self.analysed_synapse.append(synapse)
                
            elif value['type'] == 'global' and value['var'].eq != None: # A parameter with equation
                cpp_type, init_value = get_value_and_type(name, value)

                tree = Tree.Tree(self, name, value['var'].eq)
                if not tree.success: # Error while processing the equation
                    return None, None
                self.trees.append(tree)
                self.analysed_synapse.append(
                    {'name': name,
                     'type': 'global',
                     'init': self.init_parameter(name, init_value),
                     'cpp' : tree.cpp()+';',
                     'cpp_type': cpp_type
                     } )
                
            else:
                cpp_type, init_value = get_value_and_type(name, value)
                self.analysed_synapse.append(
                    {'name': name,
                     'type': 'global',
                     'init': self.init_parameter(name, init_value),
                     'cpp' : '',
                     'cpp_type': cpp_type
                     } )
                
        # Process the global operations
        self.global_operations = sort_global_operations(self.global_operations)

        return self.analysed_synapse, self.global_operations

    def init_parameter(self, name, value):
        return name + '_ = ' + str(value) + ';';

    def init_local_variable(self, name, value):
        return name +'_ = std::vector< '+DATA_TYPE+' >(pre_population_->getNeuronCount(), '+str(value)+');\n'

    def init_global_variable(self, name, value):
        return name + '_ = '+str(value)+';'

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
            if 'pre' in deps or 'value' in deps:
                sorted_dependencies[name] = True
        stable = False
        while not stable:
            stable = True
            for name, deps in dependencies.items():
                is_dep_pre = False
                for dep in deps:
                    if not dep == 'pre' and not dep == 'value':
                        is_dep_pre = is_dep_pre or sorted_dependencies[dep]
                        if not is_dep_pre == sorted_dependencies[name]:
                            stable = False
                            sorted_dependencies[name] = is_dep_pre

        localvar = ['value']
        globalvar = []
        for name in dependencies.keys():
            if name == 'value':
                continue # already sorted in

            if sorted_dependencies[name]:
                localvar.append(name)
            else:
                globalvar.append(name)
        return localvar, globalvar

def sort_global_operations(operations):
    global_operations = {'pre': [], 'post': []}
    for ope in operations['pre']:
        if not ope in global_operations['pre']: #does not already exist
            global_operations['pre'].append(ope)
    for ope in operations['post']:
        if not ope in global_operations['post']: #does not already exist
            global_operations['post'].append(ope)

    return global_operations
