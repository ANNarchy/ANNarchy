"""

    SpikePopulation.py
    
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
from ANNarchy4.core import Global
from ANNarchy4.core.Random import RandomDistribution
from Population import Population
import re
         
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
               
class SpikePopulation(Population):
    """
    Population generator class
    """
    
    def __init__(self, population):

        Population.__init__(self, population)
        
    def generate(self, verbose):
        """
        Generates the cpp codes for the spike coded population.
        """        
        self._update_neuron_variables()
            
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self._generate_header())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self._generate_body())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self._generate_pyx())

    def _generate_header(self):
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
                if value['type'] == 'rand_variable':
                    continue
    
                if value['type'] == 'variable':
                    # Array access
                    access += '    void set'+value['name'].capitalize()+'(std::vector<DATA_TYPE> '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += '    std::vector<DATA_TYPE> get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
                    # Individual access
                    access += '    void setSingle'+value['name'].capitalize()+'(int rank, DATA_TYPE '+value['name']+') { this->'+value['name']+'_[rank] = '+value['name']+'; }\n\n'
                    access += '    DATA_TYPE getSingle'+value['name'].capitalize()+'(int rank) { return this->'+value['name']+'_[rank]; }\n\n'

                    # Recording
                    access += '    std::vector< std::vector<DATA_TYPE> >getRecorded'+value['name'].capitalize()+'() { return this->recorded_'+value['name']+'_; }\n\n'                    
                    access += '    void startRecord'+value['name'].capitalize()+'() { this->record_'+value['name']+'_ = true; }\n\n'
                    access += '    void stopRecord'+value['name'].capitalize()+'() { this->record_'+value['name']+'_ = false; }\n\n' 
                    access += '    void clearRecorded'+value['name'].capitalize()+'() { this->recorded_'+value['name']+'_.clear(); }\n\n'   
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
    
                member += '\t' + value['def'] + '\n'
            
                #recording
                if value['type'] == 'variable':
                    member += '\t'+'bool record_'+value['name']+'_;\n'
                    member += '\t'+'std::vector< std::vector<DATA_TYPE> > recorded_'+value['name']+'_;\n'
            
            for value in global_ops['post']:
                member += '\tDATA_TYPE '+ value['variable']+'_'+value['function']+'_;\n\n'
                
            return member
            
        def global_ops(global_ops):
            """
            Add the computation calls for min, max, mean, etc. operations.
            """
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

    def _generate_body(self):
        """
        Create population class implementation.
        """
        
        def constructor(parsed_neuron):
            """
            generate population constructor implementation.
            """
            constructor = ''
    
            for value in parsed_neuron:

                if value['type'] != 'rand_variable':
                    constructor += '\t'+value['init']+'\n'
    
            constructor += '\tdt_ = ' + str(Global.config['dt']) + ';'
            return constructor
        
        def destructor(parsed_neuron):
            """
            generate population destructor implementation.
            """
            destructor = ''
    
            for value in parsed_neuron:
                if '_rand_' in value['name']:   # skip local member
                    continue
                if 'rate' == value['name']:   # fire rate
                    continue

                if 'variable' == value['type']:                 
                    destructor += '\t'+value['name']+'_.clear();\n'
    
            return destructor

        def record(parsed_neuron):
            """
            Generate the function body of the Population::record method.
            """
            code = ''
            
            for var in parsed_neuron:
                if '_rand_' in var['name']:
                    continue
                
                if var['type'] == 'variable':
                    code += '''\tif(record_%(var)s_)\n\t\trecorded_%(var)s_.push_back(%(var)s_);\n''' % { 'var': var['name'] }
                
            return code
        
        def single_global_ops(class_name, global_ops):
            """
            Generate the implementation for requested attributes.
            """
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
        
        def reset(parsed_neuron):
            """
            implementation of reset values.
            """
            def find_val(parsed_neuron, name):
                for var in parsed_neuron:
                    if var['name'].find(name)==0:
                        return var
                        
            loop = ''
            
            for value in parsed_neuron:
                
                #
                # TODO: maybe better in parser??
                if 'reset' in value.keys():
                    for reset_val in value['reset']:
                        var = re.findall('[A-Za-z]+', reset_val.split("=")[0])[0] # only on match exist, but spaces are supressed
                        val = reset_val.split("=")[1]
                        
                        #
                        # left side of equation
                        lside = ''
                        var_type = find_val(parsed_neuron, var)['type']
                        if var_type == 'variable':
                            lside = var+'_[(*it)]'
                        else:
                            lside = var

                        #
                        # right side
                        values = re.findall('[A-Za-z]+', val)
                        
                        for tval in values:
                            v_t = find_val(parsed_neuron, tval)['type']
                            if v_t == 'variable':
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
            for value in parsed_neuron:
                if value['type'] == 'rand_variable':
                    meta += """\t%(name)s_ = %(cpp_obj)s.getValues(nbNeurons_);\n""" % { 'name': value['name'], 'cpp_obj': value['eq']._gen_cpp() }
            
            loop = ''
    
            #
            # generate loop code
            if order == []:
                # order does not play an important role        
                for value in parsed_neuron:
                    if 'rand_variable' == value['type']:   # skip local member
                        continue
    
                    loop += '\t\t'+value['cpp']+'\n'
                    if 'min' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value['name'], 'border': value['min'] }
                    if 'max' in value.keys():
                        loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value['name'], 'border': value['max'] }
                
            else:
                for value in order:
                    for value2 in parsed_neuron:
                        if 'rand_variable' == value2['type']:   # skip local member
                            continue
    
                        if value2['name'] == value:
                            loop += '\t\t'+value2['cpp']+'\n'
    
                            if 'min' in value2.keys():
                                loop += '''\t\t if (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value2['name'], 'border': value2['min'] }
                            if 'max' in value2.keys():
                                loop += '''\t\t if (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': value2['name'], 'border': value2['max'] }

            for value in parsed_neuron:
                if 'threshold' in value.keys():
                    
                    loop += '''\t\t if (%(name)s_[i] > %(threshold)s)\n\t\t {\n\t\t\treset_.push_back(i);\n\t\t\tpropagate_.push_back(i);\n\n\t\t\tspike_timings_[i].push_back(ANNarchy_Global::time);\n\t\t\tspiked_[i] = true;\n\t\t }\n''' % { 'name': value['name'], 'threshold': value['threshold'] }
    
            code = meta
            code += '\tfor(int i=0; i<nbNeurons_; i++)\n' 
            code += '\t{\n'
            code += loop
            code += '\t}\n'
            
            return code

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

    def _generate_pyx(self):
        """
        Create population class python extension.
        """
        
        def pyx_func(parsed_neuron, global_ops):
            """
            function calls to wrap the c++ accessors.
            """
            code = ''
    
            for value in parsed_neuron:
                if value['type'] == 'rand_variable':
                    continue
    
                if value['type'] == 'variable':
                    var_code = """
        vector[%(type)s] get%(Name)s()\n
        void set%(Name)s(vector[%(type)s] values)\n
        %(type)s getSingle%(Name)s(int rank)\n
        void setSingle%(Name)s(int rank, %(type)s values)\n
        void startRecord%(Name)s()\n
        void stopRecord%(Name)s()\n
        void clearRecorded%(Name)s()\n
        vector[vector[%(type)s]] getRecorded%(Name)s()\n
""" % { 'Name': value['name'].capitalize(), 'name': value['name'], 'type': get_type_name(value['cpp_type']) }
                    
                    code += var_code.replace('DATA_TYPE', 'float')
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
                if value['type'] == 'rand_variable':
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
                    code += '    def _start_record_'+ value['name'] + '(self):\n'   
                    code += '        self.cInstance.startRecord'+value['name'].capitalize()+'()\n\n'  
                    code += '    def _stop_record_'+ value['name'] + '(self):\n'   
                    code += '        self.cInstance.stopRecord'+value['name'].capitalize()+'()\n\n'  
                    code += '    def _get_recorded_'+ value['name'] + '(self):\n'
                    code += '        tmp = np.array(self.cInstance.getRecorded'+value['name'].capitalize()+'())\n\n'
                    code += '        self.cInstance.clearRecorded'+value['name'].capitalize()+'()\n'   
                    code += '        return tmp\n\n'                      
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
        
