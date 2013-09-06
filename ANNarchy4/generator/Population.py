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
    Population generator class
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
        self.global_operations = {'pre':[],'post':[]}
        self.post_compilation_init = {}
        
    def _init_variables(self):
        """ Called after creation of the C++ objects to initialize variables with arrays."""
        self.population.set(self.post_compilation_init)
             
    def _get_value(self, name):
        """ Returns init value """
        if name in self.post_compilation_init.keys():
            return self.post_compilation_init[name]
        for var in self.neuron_variables:
            if var['name']==name:
                if 'var' in var.keys(): # variable
                    return var['var'].init
                elif 'init' in var.keys(): # parameter
                    return var['init']
                else: # default
                    return 0.0 
        return None
        
    def _variable_names(self):
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
        print "Error: it's not allowed to add new variables / parameters to population object."

    def _update_value(self, name, value):

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
                    print "Error: can't assign ", value ,"(",type(value),") to the variable "+name
            else:
                values['init'] = float(value)
        else:
            print "Error: variable / parameter "+name+" does not exist in population object."
            
    def generate(self):
        """
        main function of population generator class.
        """        
        

        #   replace all RandomDistribution by rand variables with continous
        #   numbers and stores the corresponding call as local variable
        i = 0
        for value in self.neuron_variables:
            if 'var' not in value.keys(): 
                continue 

            if value['var'].eq == None: 
                continue 

            # check, if a RD object in the equation
            if value['var'].eq.find('RandomDistribution') != -1: 
                eq_split = value['var'].eq.split('RandomDistribution')
                
                for tmp in eq_split:
                    if tmp.find('sum(') != -1:
                        continue

                    # phrase contains an array of
                    # shortest terms within two brackets
                    phrase = re.findall('\(.*?\)', tmp) 

                    if phrase != []:
                        for part in phrase:
                            self.neuron_variables.append( 
                                { 'name': '_rand_'+str(i), 
                                  'var': Variable(init = part) 
                                }
                            )
                            
                            value['var'].eq = value['var'].eq.replace(
                                'RandomDistribution'+part, '_rand_'+str(i)
                            )
                            
                            self.rand_objects.append(part)
                        i += 1

        #   parse neuron
        self.neuron_parser = parser.NeuronAnalyser(
            self.neuron_variables, 
            self.targets
        )
        self.parsed_neuron, global_operations = self.neuron_parser.parse()
        
        for g_op in global_operations['post']:
            self._add_global_oparation(g_op)

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

    def add_target(self, target):
        """
        Internal Function
        """
        self.targets.append(target)

        self.targets = list(set(self.targets))   # little trick to remove doubled entries
            
    def generate_header(self):
        """
        Create population class header.
        """
        
        def generate_accessor(neuron_values, global_ops):
            """
            Creates for all variables/parameters of the neuron the 
            corresponding set and get methods. 
            """
            access = ''
    
            for value in neuron_values:
                if '_rand_' in value['name']:
                    continue
    
                if value['type'] == 'variable':
                    # Array access
                    access += '    void set'+value['name'].capitalize()+'(std::vector<DATA_TYPE> '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += '    std::vector<DATA_TYPE> get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
                    # Individual access
                    access += '    void setSingle'+value['name'].capitalize()+'(int rank, DATA_TYPE '+value['name']+') { this->'+value['name']+'_[rank] = '+value['name']+'; }\n\n'
                    access += '    DATA_TYPE getSingle'+value['name'].capitalize()+'(int rank) { return this->'+value['name']+'_[rank]; }\n\n'
                else:
                    access += '    void set'+value['name'].capitalize()+'(DATA_TYPE '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += '    DATA_TYPE get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
    
            for value in global_ops['post']:
                access += '\tDATA_TYPE get'+value['function'].capitalize()+value['variable'].capitalize()+'() { return '+value['variable']+'_'+value['function']+'_; }\n\n'
    
            return access
    
        def generate_member_definition(neuron_values, global_ops):
            """
            Creates for all variables/parameters of the neuron the 
            corresponding definitions within the c++ class. 
            """
            member = ''
            
            for value in neuron_values:
                if '_rand_' in value['name']:   # skip local member
                    member += '\tstd::vector<DATA_TYPE> '+value['name']+'_;\n'
                    continue
                if 'rate' == value['name']:
                    continue
    
                member += '\t'+value['def']+'\n'
            
            for value in global_ops['post']:
                member += '\tDATA_TYPE '+ value['variable']+'_'+value['function']+'_;\n\n'
                
            return member
            
        def global_ops(global_ops):
            code = ''
            
            for var in global_ops['post']:
                code += '\tvoid compute_'+var['function']+'_'+var['variable']+'();\n\n'
                
            return code
            
        header = """#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    int getNeuronCount() { return nbNeurons_; }
    
    void metaStep();
    
    void globalOperations();
    
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
    
            for value in parsed_neuron:
                if '_rand_' in value['name']:   # skip local member
                    continue
                    
                constructor += '\t'+value['init']+'\n'
    
            constructor += '\tdt_ = ' + str(Global.config['dt']) + ';'
            return constructor
        
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
                    code+="""void %(class)s::compute_min_%(var)s() {
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
            
            meta = replace_rand_(parsed_neuron, rand_objects)
            loop = ''
    
            #
            # generate loop code
            if order == []:
                # order does not play an important role        
                for value in parsed_neuron:
                    if '_rand_' in value['name']:   # skip local member
                        continue
    
                    loop += '\t\t'+value['cpp']+'\n'
                    if 'min' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value['name'], 'border': value['min'] }
                    if 'max' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value['name'], 'border': value['max'] }
                    
                
            else:
                for value in order:
                    for value2 in parsed_neuron:
                        if '_rand_' in value2['name']:   # skip local member
                            continue
    
                        if value2['name'] == value:
                            loop += '\t\t'+value2['cpp']+'\n'
    
                            if 'min' in value2.keys():
                                loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value2['name'], 'border': value2['min'] }
                            if 'max' in value2.keys():
                                loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value2['name'], 'border': value2['max'] }
    
            code = meta
            code += '\tfor(int i=0; i<nbNeurons_; i++) {\n'
            code += loop
            code += '}\n'
            
            return code

        body = """#include "%(class)s.h"

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(construct)s
    Network::instance()->addPopulation(this);
}

void %(class)s::metaStep() {
%(metaStep)s    
}

void %(class)s::globalOperations() {
%(global_ops)s
}

%(single_global_ops)s
""" % { 'class': self.class_name, 
        'construct': constructor(self.parsed_neuron), 
        'metaStep': meta_step(self.parsed_neuron,
                              self.rand_objects,
                              self.population.neuron_type.order
                              ),
        'global_ops': global_ops(self.global_operations),
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
    
            for value in parsed_neuron:
                if '_rand_' in value['name']:
                    continue
    
                if value['type'] == 'variable':
                    code += '        vector[float] get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'(vector[float] values)\n\n'
                    code += '        float getSingle'+value['name'].capitalize()+'(int rank)\n\n'
                    code += '        void setSingle'+value['name'].capitalize()+'(int rank, float values)\n\n'
                else:
                    code += '        float get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'(float value)\n\n'
    
            return code
    
        def py_func(parsed_neuron, global_ops):
            """
            function calls to provide access from python
            to c++ data.
            """            
            code = ''
    
            for value in parsed_neuron:
                if '_rand_' in value['name']:
                    continue
    
                code += '    property '+value['name']+':\n'
                if value['type'] == 'variable':
                    code += '        def __get__(self):\n'
                    code += '            return np.array(self.cInstance.get'+value['name'].capitalize()+'())\n\n'
    
                    code += '        def __set__(self, value):\n'
                    code += '            if isinstance(value, np.ndarray)==True:\n'
                    code += '                if value.ndim==1:\n'
                    code += '                    self.cInstance.set'+value['name'].capitalize()+'(value)\n'
                    code += '                else:\n'
                    code += '                    self.cInstance.set'+value['name'].capitalize()+'(value.reshape(self.size))\n'

                    code += '            else:\n'
                    code += '                self.cInstance.set'+value['name'].capitalize()+'(np.ones(self.size)*value)\n\n'  
                    code += '    def _get_single_'+ value['name'] + '(self, rank):\n'   
                    code += '        return self.cInstance.getSingle'+value['name'].capitalize()+'(rank)\n\n'  
                    code += '    def _set_single_'+ value['name'] + '(self, rank, value):\n'   
                    code += '        self.cInstance.setSingle'+value['name'].capitalize()+'(rank, value)\n\n' 
                else:
                    code += '        def __get__(self):\n'
                    code += '            return self.cInstance.get'+value['name'].capitalize()+'()\n\n'
                
                    code += '        def __set__(self, value):\n'
                    code += '            self.cInstance.set'+value['name'].capitalize()+'(value)\n'
    
                
            return code
        
        pyx = '''from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
%(cFunction)s


cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self):
        self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

    def name(self):
        return self.cInstance.getName()

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
        return pyx

        
