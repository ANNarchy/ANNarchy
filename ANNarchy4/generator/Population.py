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
import re
import numpy as np

# ANNarchy core informations
import ANNarchy4.core.Global as Global
from ANNarchy4.core.Random import RandomDistribution

from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Neuron import RateNeuron
from ANNarchy4 import parser

import copy

def get_type_name(type):
    """
    mapping between python types and cpp names, float will be mapped to
    DATA_TYPE to allow changing between precisions in the framework.
    """
    if type==float:
        return 'DATA_TYPE'
    elif type==int:
        return 'int'
    elif type==bool:
        return 'bool'
    else:
        print "Unknown type, use default = 'DATA_TYPE'"
        return 'DATA_TYPE'

class Population(object):
    """
    Population generator base class
    """
    def __init__(self, population):
        self.class_name = 'Population'+str(population.rank)
        
        self.header = Global.annarchy_dir+'/generate/build/'+self.class_name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.class_name+'.pyx'
        
        self.population = population
        
        self.rand_objects = []
        self.targets = []
        self.neuron_variables = copy.deepcopy(self.population.neuron_type.variables)
        self.global_operations = { 'pre':[], 'post':[] }
        self.post_compilation_init = {}
        
    def _init_variables(self):
        """ Called after creation of the C++ objects to initialize variables with arrays."""
        self.population.set(self.post_compilation_init)

    def _add_target(self, target):
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
        return self.neuron_variables.keys()

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
        
        try:
            values = self.neuron_variables[name]
            if 'var' in values.keys():
                if isinstance(value, (int, float)):       
                    values['var'].init = float(value)
                elif isinstance(value, Variable):
                    values['var'] = value
                elif isinstance(value, RandomDistribution):
                    if isinstance(values['var'].eq, RandomDistribution):
                        values['var'].eq = value
                    else:
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
        
        except KeyError:
            print "Error: variable / parameter "+name+" does not exist in population object."
            
    def _update_neuron_variables(self):
        """
        Updates the neuron variable dictionary e.g. :
            
            * global operations
            
        Should be called before code generation starts.
        """
        #   parse neuron
        self.neuron_parser = parser.NeuronAnalyser(
            self.neuron_variables, 
            self.targets,
            self.population.name
        )
        
        self.parsed_neuron, global_operations = self.neuron_parser.parse()
        
        #   attach needed global operations e.g. min, max, mean
        for g_op in global_operations['post']:
            self._add_global_oparation(g_op)       
            
    def generate(self, verbose):
        """
        Generate the population class based on neuron definition.
        """        
        if verbose:
            if self.class_name != self.population.name:
                print '    for', self.class_name, '(', self.population.name, ')' 
            else:
                print '    for', self.class_name

        self._update_neuron_variables()

        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self.generate_header())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self.generate_body())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generate_pyx()) 
            
    def generate_cpp_add(self):
        """
        generates population instantiation code for c++. 
        HINT: only used if cpp_stand_alone=True provided to generator.compile()
        """        
        return ('\t\t'+self.population.generator.class_name+'* '+
                      self.population.name+' = new '+
                      self.population.generator.class_name+'("'+
                      self.population.name+'", '+
                      str(self.population.size)+');\n')
                       
    def generate_header(self):
        """
        Create population class header.
        """
    
        def generate_accessor(neuron_values, global_ops):
            """
            Creates for all variables/parameters of the object the corresponding set and get methods.
            """
            access = ''
    
            for name, value in neuron_values.iteritems():
                #
                # variables
                if value['type'] == 'local':
                    # Array access
                    access += """
\tstd::vector<%(type)s> get%(Name)s() { return this->%(name)s_; }
\tvoid set%(Name)s(std::vector<%(type)s> %(name)s) { this->%(name)s_ = %(name)s; }
""" % { 'Name': name.capitalize(), 'name': name, 'type': get_type_name(value['cpp_type'])}
    
                    # Individual access
                    access += """
\t%(type)s getSingle%(Name)s(int rank) { return this->%(name)s_[rank]; }
\tvoid setSingle%(Name)s(int rank, %(type)s %(name)s) { this->%(name)s_[rank] = %(name)s; }
""" % { 'Name': name.capitalize(), 'name': name, 'type': get_type_name(value['cpp_type'])}
                    
                    # Recording
                    access += """
\tstd::vector< std::vector< %(type)s > >getRecorded%(Name)s() { return this->recorded_%(name)s_; }                    
\tvoid startRecord%(Name)s() { this->record_%(name)s_ = true; }
\tvoid stopRecord%(Name)s() { this->record_%(name)s_ = false; }
\tvoid clearRecorded%(Name)s() { this->recorded_%(name)s_.clear(); }
""" % { 'Name': name.capitalize(), 'name': name, 'type': get_type_name(value['cpp_type'])}
                
                #
                # parameters
                elif value['type'] == 'global':
                    access += """
\t%(type)s get%(Name)s() { return this->%(name)s_; }
\tvoid set%(Name)s(%(type)s %(name)s) { this->%(name)s_ = %(name)s; }
""" % { 'Name': name.capitalize(), 'name': name, 'type': get_type_name(value['cpp_type'])}
    
                #
                # rand variable
                else:
                    continue
    
            for value in global_ops['post']:
                access += '\tDATA_TYPE get'+value['function'].capitalize()+value['variable'].capitalize()+'() { return '+value['variable']+'_'+value['function']+'_; }\n\n'
    
            return access
        
        def generate_member_definition(neuron_values, global_ops):
            """
            Creates for all variables/parameters of the neuron the 
            corresponding definitions within the c++ class. 
            """
            member = ''
            
            for name, value in neuron_values.iteritems():
                if name == 'rate':
                    # rate recording
                    member += '\t'+'bool record_rate_;\n'
                    member += '\t'+'std::vector< std::vector<DATA_TYPE> > recorded_rate_;\n'                    
    
                elif value['type'] == 'local':
                    member += """
\tstd::vector<%(type)s> %(name)s_;
\tbool record_%(name)s_; 
\tstd::vector< std::vector<%(type)s> > recorded_%(name)s_;\n
""" % { 'name': name, 'type': get_type_name(value['cpp_type']) }

                elif value['type'] == 'rand_variable':
                    member += """
\tstd::vector<%(type)s> %(name)s_;\n
""" % { 'name': name, 'type': get_type_name(value['cpp_type']) }
                    
                else:
                    member += """
\t%(type)s %(name)s_;\n
""" % { 'name': name, 'type': get_type_name(value['cpp_type']) }
                    
            for value in global_ops['post']:
                member += '\tDATA_TYPE '+ value['variable']+'_'+value['function']+'_;\n\n'

                
            return member
            
        def global_ops(global_ops):
            code = ''
            
            for var in global_ops['post']:
                code += '\tvoid compute_'+var['function']+'_'+var['variable']+'();\n\n'
                
            return code
        
        if isinstance(self.population.neuron_type, RateNeuron):    
            header = """#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit();
    
    void metaStep();
    
    void globalOperations();
    
    void record();
    
%(access)s
private:
%(global_ops)s

%(member)s
};
#endif
""" % { 'class': self.class_name, 
        'member': generate_member_definition(self.parsed_neuron, 
                                             self.global_operations),
        'global_ops': global_ops(self.global_operations), 
        'access' : generate_accessor(self.parsed_neuron,
                                     self.global_operations) 
        } 
        else:
           header = """#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void metaStep();
    
    void globalOperations();
    
    void propagateSpike();
    
    void reset();
    
    void record();
    
%(access)s
private:
%(global_ops)s

%(member)s

    std::vector<int> propagate_;    ///< neurons which will propagate their spike
    std::vector<int> reset_;    ///< neurons which will reset after current eval
};
#endif
""" % { 'class': self.class_name, 
        'member': generate_member_definition(self.parsed_neuron, 
                                             self.global_operations),
        'global_ops': global_ops(self.global_operations), 
        'access' : generate_accessor(self.parsed_neuron,
                                     self.global_operations) 
        }
 
        return header

    def generate_body(self):
        """
        Create population class implementation.
        """
                    
        def constructor(parsed_neuron):
            """
            generate population constructor implementation.
            """
            constructor = ''
    
            for name, value in parsed_neuron.iteritems():
                curr_line = ''

                if value['type'] == 'rand_variable':
                    constructor += """\t%(name)s_ = %(cpp_obj)s.getValues(nbNeurons_);\n""" % { 'name' : name, 'cpp_obj': value['eq']._gen_cpp() }

                else:
                    if 'True' in value['init']:
                        curr_line += '\t'+value['init'].replace('True',str(1.0))
                    elif 'False' in value['init']:
                        curr_line += '\t'+value['init'].replace('False',str(0.0))
                    else:
                        curr_line += '\t'+value['init']
    
                    if value['cpp_type'] != float:
                        curr_line = curr_line.replace("DATA_TYPE", get_type_name(value['cpp_type']))               
    
                    if value['type'] == 'variable':
                        curr_line += """
\trecord_%(name)s_ = false;
\trecorded_%(name)s_ = std::vector< std::vector< %(type)s > >();
""" % { 'name': name, 'type': get_type_name(value['cpp_type']) }

                constructor += curr_line + '\n'
    
            constructor += '\tdt_ = ' + str(Global.config['dt']) + ';'
            return constructor
        
        def destructor(parsed_neuron):
            """
            generate population destructor implementation.
            """
            destructor = ''
    
            for name, value in parsed_neuron.iteritems():
                if name == 'rate':   # fire rate
                    continue

                if value['type'] == 'local':
                    destructor += '\t'+name+'_.clear();\n'
    
            return destructor
        
        def record(parsed_neuron):
            """
            Generate the function body of the Population::record method.
            """            
            code = ''
            
            for name, value in parsed_neuron.iteritems():
                if value['type'] == 'rand_variable':
                    continue
                
                if value['type'] == 'local':
                    code += '''\tif(record_%(var)s_)\n\t\trecorded_%(var)s_.push_back(%(var)s_);\n''' % { 'var': name }
                
            return code
            
        def replace_rand_(parsed_neuron, rand_objects):
            """
            Before the equations are provided to the parser all RandomDistribution
            calls are masked by unique names. Now we revert this masking.
            """
            meta_code = ''
            
            #
            # revert the replace of RD objects
            for value in parsed_neuron:
                if '_rand_' in value['name']:
                    
                    idx = int(value['name'].split('_rand_')[1])
                    parameters = rand_objects[idx]
                    call = 'RandomDistribution' + parameters + '.genCPP()'
                    try:
                        random_cpp = eval(call)
                    except Exception, exception:
                        print exception
                        print 'Error in', value['eq'], ': the RandomDistribution object is not correctly defined.'
                        exit(0) 
                    meta_code += '\t'+value['name']+'_= '+random_cpp+'.getValues(nbNeurons_);\n'
                    
            return meta_code
        
        def single_global_ops(class_name, global_ops):
            code = ''
            
            for var in global_ops['post']:
                if var['function'] == 'min':
                    code += """void %(class)s::compute_min_%(var)s() {
    %(var)s_min_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        if(%(var)s_[i] < %(var)s_min_)
            %(var)s_min_ = %(var)s_[i];
    }
}\n\n""" % { 'class': class_name, 'var': var['variable'] }
                elif var['function'] == 'max':
                    code += """void %(class)s::compute_max_%(var)s() {
    %(var)s_max_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        if(%(var)s_[i] > %(var)s_max_)
            %(var)s_max_ = %(var)s_[i];
    }
}\n\n"""  % { 'class': class_name, 'var': var['variable'] }
                elif var['function'] == 'mean':
                    code += """void %(class)s::compute_mean_%(var)s() {
    %(var)s_mean_ = %(var)s_[0];
    for(unsigned int i=1; i<%(var)s_.size();i++){
        %(var)s_mean_ += %(var)s_[i];
    }
    
    %(var)s_mean_ /= %(var)s_.size();
}\n\n"""  % { 'class': class_name, 'var': var['variable'] }
                else:
                    print "Error: unknown operation - '"+var['function']+"'"
                
            return code
        
        def global_ops(global_ops):
            code = ''
            
            for variable in global_ops['post']:
                code += '\tcompute_'+variable['function']+'_'+variable['variable']+'();'
                
            return code
        
        def meta_step(parsed_neuron, rand_objects, order):
            """
            Parallel evaluation of neuron equations.
            """
            
            meta = ''
            # initialize the random variables before the loop
            for name, value in parsed_neuron.iteritems():
                if value['type'] == 'rand_variable':
                    meta += """\t%(name)s_ = %(cpp_obj)s.getValues(nbNeurons_);\n""" % { 'name': name, 'cpp_obj': value['eq']._gen_cpp() }
            
            loop = ''
    
            #
            # generate loop code
            for var_name in order:
                value = parsed_neuron[var_name]
                if value['type'] == 'local':

                    loop += '\t\t'+value['cpp']+'\n'
    
                    if 'min' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var_name, 'border': value['min'] }
                    if 'max' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var_name, 'border': value['max'] }

            for name, value in parsed_neuron.iteritems():
                if 'threshold' in value.keys():
                    try:
                        th = parsed_neuron[value['threshold'].replace(' ','')]
                        if th['type']=='local':
                            var = value['threshold']+'_[i]'
                        else:
                            var = value['threshold']
                    except KeyError:
                        var = value['threshold']
                        
                    loop += '''\t\t if (%(name)s_[i] > %(threshold)s)\n\t\t {\n\t\t\treset_.push_back(i);\n\t\t\tpropagate_.push_back(i);\n\n\t\t\tspike_timings_[i].push_back(ANNarchy_Global::time);\n\t\t\tspiked_[i] = true;\n\t\t }\n''' % { 'name': name, 'threshold': var }
    
            code = meta + '\n'
            code += '\tfor(int i=0; i<nbNeurons_; i++)\n' 
            code += '\t{\n'
            code += loop
            code += '\t}\n'

            code += '#ifdef _DEBUG\n'
            code += '\tstd::cout << "Population:" << name_ << std::endl;\n'
            code += '\tstd::cout << "before:" << std::endl;\n'
            code += '\tfor (auto it=delayedRates_.begin(); it!=delayedRates_.end(); it++)\n'
            code += '\t{\n'
            code += '\t\tstd::cout << "rates: "<< std::endl;\n'
            code += '\t\tfor (auto it2 = (*it).begin(); it2 != (*it).end(); it2++)\n'
            code += '\t\t{\n'
            code += '\t\t\tstd::cout << (*it2) << " ";\n'
            code += '\t\t}\n'
            code += '\t\tstd::cout << std::endl;\n'
            code += '\t}\n'
            
            code += '\tstd::cout << "maxDelay:" << maxDelay_ << std::endl;\n'
            code += '\tif (maxDelay_ > 1)'
            code += '\t{\n'
            code += '\t\tdelayedRates_.push_front(rate_);\n'
            code += '\t\tdelayedRates_.pop_back();\n'
            code += '\t}\n'

            code += '\tstd::cout << "after:" << std::endl;'
            code += '\tfor (auto it=delayedRates_.begin(); it!=delayedRates_.end(); it++)\n'
            code += '\t{\n'
            code += '\t\tstd::cout << "rates: "<< std::endl;\n'
            code += '\t\tfor (auto it2 = (*it).begin(); it2 != (*it).end(); it2++)\n'
            code += '\t\t{\n'
            code += '\t\t\tstd::cout << (*it2) << " ";\n'
            code += '\t\t}\n'
            code += '\t\tstd::cout << std::endl;\n'
            code += '\t}\n'
            code += '#endif'
            
            return code

        def reset(parsed_neuron):
            """
            implementation of reset values after a spike was emited.
            """
            loop = ''
            
            for name, value in parsed_neuron.iteritems():
                
                #
                # TODO: maybe better in parser??
                if 'reset' in value.keys():
                    for reset_val in value['reset']:
                        var = re.findall('[A-Za-z]+', reset_val.split("=")[0])[0] # only on match exist, but spaces are supressed
                        val = reset_val.split("=")[1]
                        
                        #
                        # left side of equation
                        lside = ''
                        var_type = parsed_neuron[var]['type']
                        if var_type == 'local':
                            lside = var+'_[(*it)]'
                        else:
                            lside = var

                        #
                        # right side
                        values = re.findall('[A-Za-z]+', val)
                        
                        for tval in values:
                            v_t = parsed_neuron[tval]['type']
                            if v_t == 'local':
                                val = val.replace(tval, tval+'_[(*it)]')
                            else:
                                val = val.replace(tval, tval+'_')
                                
                        loop +='''\t\t\t%(lside)s = %(rside)s;\n''' % { 'lside': lside, 'rside': val }

            code = """
        for(auto it=reset_.begin(); it != reset_.end(); it++) 
        {
%(loop)s
        } """ % { 'loop': loop }

            return code
        
        def reset_to_init(parsed_neuron):
            """
            generate code for reset to initial values.
            """
            code = ''

            for name, value in parsed_neuron.iteritems():
                if value['type'] == 'rand_variable':
                    code += """\t%(name)s_ = %(cpp_obj)s.getValues(nbNeurons_);\n""" % { 'name': name, 'cpp_obj': value['eq']._gen_cpp() }

                elif value['type'] == 'local':

                    curr_line = ''
                    if 'True' in value['init']:
                        curr_line += '\t'+value['init'].replace('True',str(1.0))+'\n'
                    elif 'False' in value['init']:
                        curr_line += '\t'+value['init'].replace('False',str(0.0))+'\n'
                    else:
                        curr_line += '\t'+value['init']+'\n'
                        
                    if value['cpp_type'] != float:
                        curr_line = curr_line.replace("DATA_TYPE", get_type_name(value['cpp_type']))
                    
                    code += curr_line
                
            return code

        if isinstance(self.population.neuron_type, RateNeuron):
            body = """#include "%(class)s.h"
#include "Global.h"
using namespace ANNarchy_Global;

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(construct)s
    Network::instance()->addPopulation(this);
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif

    std::cout << "%(class)s::Destructor" << std::endl;
    
%(destruct)s
}

void %(class)s::resetToInit() {
%(resetToInit)s
}

void %(class)s::metaStep() {
%(metaStep)s    
}

void %(class)s::globalOperations() {
%(global_ops)s
}

void %(class)s::record() {
%(record)s
}

%(single_global_ops)s
""" % { 'class': self.class_name, 
        'construct': constructor(self.parsed_neuron),
        'destruct': destructor(self.parsed_neuron), 
        'metaStep': meta_step(self.parsed_neuron,
                              self.rand_objects,
                              self.population.neuron_type.order
                              ),
        'resetToInit': reset_to_init(self.parsed_neuron),
        'global_ops': global_ops(self.global_operations),
        'record' : record(self.parsed_neuron),
        'single_global_ops': single_global_ops(self.class_name, self.global_operations)
    } 
        else:
            body = """#include "%(class)s.h"
#include "Global.h"
using namespace ANNarchy_Global;

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(construct)s
    Network::instance()->addPopulation(this);
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif

    std::cout << "%(class)s::Destructor" << std::endl;
    
%(destruct)s
}

void %(class)s::metaStep() {
    spiked_ = std::vector<bool>(nbNeurons_, false);
%(metaStep)s    
}

void %(class)s::propagateSpike() {

    if (!propagate_.empty())
    {

        propagate_.erase(propagate_.begin(), propagate_.end());
    }
        
}

void %(class)s::reset() {

    if (!reset_.empty())
    {
%(reset)s

        reset_.erase(reset_.begin(), reset_.end());
    }
    
}

void %(class)s::globalOperations() 
{

    propagateSpike();
    
    reset();
    
%(global_ops)s
}

void %(class)s::record() 
{
%(record)s
}

%(single_global_ops)s
""" % { 'class': self.class_name, 
        'construct': constructor(self.parsed_neuron),
        'destruct': destructor(self.parsed_neuron), 
        'metaStep': meta_step(self.parsed_neuron,
                              self.rand_objects,
                              self.population.neuron_type.order
                              ),
        'reset': reset(self.parsed_neuron),
        'global_ops': global_ops(self.global_operations),
        'record' : record(self.parsed_neuron),
        'single_global_ops': single_global_ops(self.class_name, self.global_operations)
    }
        return body    

    def generate_pyx(self):
        """
        Create population class python extension.
        """
        
        def pyx_func(parsed_neuron, global_ops):
            """
            function calls to wrap the c++ accessors.
            """
            code = ''
    
            for name, value in parsed_neuron.iteritems():
                if value['type'] == 'rand_variable':
                    continue
    
                if value['type'] == 'local':
                    var_code = """
        vector[%(type)s] get%(Name)s()\n
        void set%(Name)s(vector[%(type)s] values)\n
        %(type)s getSingle%(Name)s(int rank)\n
        void setSingle%(Name)s(int rank, %(type)s values)\n
        void startRecord%(Name)s()\n
        void stopRecord%(Name)s()\n
        void clearRecorded%(Name)s()\n
        vector[vector[%(type)s]] getRecorded%(Name)s()\n
""" % { 'Name': name.capitalize(), 'name': name, 'type': get_type_name(value['cpp_type']) }
                    
                    code += var_code.replace('DATA_TYPE', 'float')
                else:
                    code += '        float get'+name.capitalize()+'()\n\n'
                    code += '        void set'+name.capitalize()+'(float value)\n\n'
    
            return code
    
        def py_func(parsed_neuron, global_ops):
            """
            function calls to provide access from python
            to c++ data.
            """            
            code = ''
    
            for name, value in parsed_neuron.iteritems():
                if value['type'] == 'rand_variable':
                    continue
    
                code += '    property '+name+':\n'
                
                if value['type'] == 'local':
                    code += '        def __get__(self):\n'
                    code += '            return np.array(self.cInstance.get'+name.capitalize()+'())\n\n'
    
                    code += '        def __set__(self, value):\n'
                    code += '            if isinstance(value, np.ndarray)==True:\n'
                    code += '                if value.ndim==1:\n'
                    code += '                    self.cInstance.set'+name.capitalize()+'(value)\n'
                    code += '                else:\n'
                    code += '                    self.cInstance.set'+name.capitalize()+'(value.reshape(self.size))\n'

                    code += '            else:\n'
                    code += '                self.cInstance.set'+name.capitalize()+'(np.ones(self.size)*value)\n\n'  
                    code += '    def _get_single_'+ name + '(self, rank):\n'   
                    code += '        return self.cInstance.getSingle'+name.capitalize()+'(rank)\n\n'  
                    code += '    def _set_single_'+ name + '(self, rank, value):\n'   
                    code += '        self.cInstance.setSingle'+name.capitalize()+'(rank, value)\n\n' 
                    code += '    def _start_record_'+ name + '(self):\n'   
                    code += '        self.cInstance.startRecord'+name.capitalize()+'()\n\n'  
                    code += '    def _stop_record_'+ name + '(self):\n'   
                    code += '        self.cInstance.stopRecord'+name.capitalize()+'()\n\n'  
                    code += '    def _get_recorded_'+ name + '(self):\n'
                    code += '        tmp = np.array(self.cInstance.getRecorded'+name.capitalize()+'())\n\n'
                    code += '        self.cInstance.clearRecorded'+name.capitalize()+'()\n'   
                    code += '        return tmp\n\n'  
                else:
                    code += '        def __get__(self):\n'
                    code += '            return self.cInstance.get'+name.capitalize()+'()\n\n'
                
                    code += '        def __set__(self, value):\n'
                    code += '            self.cInstance.set'+name.capitalize()+'(value)\n'
    
                
            return code
        
        if isinstance(self.population.neuron_type, RateNeuron):
            pyx = '''from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        void resetToInit()
        
        void setMaxDelay(int)

%(cFunction)s


cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self):
        self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

    def name(self):
        return self.cInstance.getName()

    def reset(self):
        self.cInstance.resetToInit()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "py%(name)s.size is a read-only attribute."
            
%(pyFunction)s

''' % { 'name': self.class_name,  
        'neuron_count': self.population.size, 
        'cFunction': pyx_func(self.parsed_neuron,
                              self.global_operations), 
        'pyFunction': py_func(self.parsed_neuron,
                              self.global_operations) 
    }
        else:
            pyx = '''from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        vector[vector[int]] getSpikeTimings()
        
        void setMaxDelay(int)
        
%(cFunction)s


cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self):
        self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

    def name(self):
        return self.cInstance.getName()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "py%(name)s.size is a read-only attribute."
            
    def get_spike_timings(self):
        return np.array(self.cInstance.getSpikeTimings())
        
%(pyFunction)s

''' % { 'name': self.class_name,  
        'neuron_count': self.population.size, 
        'cFunction': pyx_func(self.parsed_neuron,
                              self.global_operations), 
        'pyFunction': py_func(self.parsed_neuron,
                              self.global_operations) 
    }
        return pyx