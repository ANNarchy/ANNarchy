"""
Population.py
"""
import re

# ANNarchy core informations
import ANNarchy4.core.Global as Global
import ANNarchy4.core.Random as ANNarRandom

from ANNarchy4.core.Variable import Variable
from ANNarchy4 import parser


class Population(object):
    """
    Population generator class
    """
    
    def __init__(self, population):
        self.header = Global.annarchy_dir+'/build/'+population.name+'.h'
        self.body = Global.annarchy_dir+'/build/'+population.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/pyx/'+population.name+'.pyx'
        
        self.population = population
        
        self.rand_objects = []
        self.neuron_variables = []
        self.targets = []
                
    def generate(self):
        """
        main function of population generator class.
        """
        
        print '\tGenerate files for '+ self.__class__.__name__
        self.neuron_variables = self.population.neuron.variables

        #
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

        #
        #   parse neuron
        self.neuron_parser = parser.NeuronAnalyser(
            self.neuron_variables, 
            self.targets
        )
        self.parsed_neuron = self.neuron_parser.parse()

        #
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
        return ('\t\t'+self.population.name.capitalize()+'* '+
                      self.population.name.lower()+' = new '+
                      self.population.name.capitalize()+'("'+
                      self.population.name.capitalize()+'", '+
                      str(self.population.size())+');\n')

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
        
        def generate_accessor(neuron_values):
            """
            Creates for all variables/parameters of the neuron the 
            corresponding set and get methods. 
            """
            access = ''
    
            for value in neuron_values:
                if '_rand_' in value['name']:
                    continue
    
                if value['type'] == 'variable':
                    #getter
                    access += '\tvoid set'+value['name'].capitalize()+'(std::vector<DATA_TYPE> '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n'
                    access += '\tstd::vector<DATA_TYPE> get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n'
                else:
                    #setter
                    access += '\tvoid set'+value['name'].capitalize()+'(DATA_TYPE '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n'
                    access += '\tDATA_TYPE get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n'
    
            return access
    
        def generate_member_definition(neuron_values):
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
            
            return member
                
        header = """#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    void metaStep();
    
%(access)s
private:
%(member)s
};
#endif
""" % { 'class': self.population.name, 
        'member': generate_member_definition(self.parsed_neuron), 
        'access' : generate_accessor(self.parsed_neuron) 
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
                    
                if 'rate' == value['name']:
                    continue
    
                constructor += '\t'+value['init']+'\n'
    
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
                    call = 'ANNarRandom.RandomDistribution' + parameters + '.genCPP()'
                    try:
                        random_cpp = eval(call)
                    except Exception, exception:
                        print exception
                        print 'Error in', value['eq'], ': the RandomDistribution object is not correctly defined.'
                        exit(0) 
                    meta_code += '\t'+value['name']+'_= '+random_cpp+'.getValues(nbNeurons_);\n'
                    
            return meta_code
        
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
                
            else:
                for value in order:
                    for value2 in parsed_neuron:
                        if '_rand_' in value2['name']:   # skip local member
                            continue
    
                        if value2['name'] == value:
                            loop += '\t\t'+value2['cpp']+'\n'
    
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
""" % { 'class': self.population.name, 
        'construct': constructor(self.parsed_neuron), 
        'metaStep': meta_step(self.parsed_neuron,
                              self.rand_objects,
                              self.population.neuron.order
                              ) 
    } 

        return body    

    def generate_pyx(self):
        """
        Create population class python extension.
        """
        
        def pyx_func(parsed_neuron):
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
                else:
                    code += '        float get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'(float value)\n\n'
    
            return code
    
        def py_func(parsed_neuron):
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
                    #getter
                    code += '        def __get__(self):\n'
                    code += '            return np.array(self.cInstance.get'+value['name'].capitalize()+'())\n\n'
    
                else:
                    #getter
                    code += '        def __get__(self):\n'
                    code += '            return self.cInstance.get'+value['name'].capitalize()+'()\n\n'
                
                code += '        def __set__(self, value):\n'
                code += '            self.cInstance.set'+value['name'].capitalize()+'(value)\n'
    
            return code
        
        pyx = '''from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/Input.h":
    cdef cppclass %(name)s:
        %(name)s(string name, int N)

%(cFunction)s
        string getName()

cdef class py%(cname)s:

    cdef %(name)s* cInstance

    def __cinit__(self):
        print 'Instantiate %(name)s'
        self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

%(pyFunction)s
    def name(self):
        return self.cInstance.getName()

''' % { 'name':self.population.name, 
        'cname':self.population.name, 
        'neuron_count': self.population.size, 
        'cFunction': pyx_func(self.parsed_neuron), 
        'pyFunction': py_func(self.parsed_neuron) 
    }
        return pyx

        
