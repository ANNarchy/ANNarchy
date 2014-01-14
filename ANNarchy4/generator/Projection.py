"""

    Projection.py
    
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
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Random import RandomDistribution
from ANNarchy4 import parser

import copy

class Projection(object):
    """
    Projection generator class.
    """
    def __init__(self, projection, synapse):
        """
        Projection generator class constructor.
        """
        self.synapse = synapse
        
        if self.synapse:
            self.synapse_variables = copy.deepcopy(synapse.variables)
        else:
            self.synapse_variables = []
            
        self.projection = projection
        self.name = projection.name

        self.h_file = Global.annarchy_dir+'/generate/build/'+self.name+'.h'
        self.cpp_file = Global.annarchy_dir+'/generate/build/'+self.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.name+'.pyx'

        self.proj_class = { 
            'ID': len(Global._projections), 
            'name': self.name 
        }
        self.global_operations = []

    def _get_value(self, name):
        """ Returns init value """
        for var in self.synapse_variables:
            if var['name']==name:
                return var['init']
        
        return None

    def _variable_names(self):
        return [var['name'] for var in self.synapse_variables ]

    def _add_value(self, name, value):
        print "Error: it's not allowed to add new variables / parameters to projection object."

    def _update_value(self, name, value):

        values = next(( item for item in self.synapse_variables if item['name']==name ), None)
        
        if values:
            if 'var' in values.keys():
                if isinstance(value, (int, float)):                
                    values['var'].init = float(value)
                elif isinstance(value, Variable):
                    values['var'] = value
                elif isinstance(value, RandomDistribution):
                    values['var'].init = value
                else:
                    print "Error: can't assign ", value ,"(",type(value),") to "+name
            else:
                values['init'] = float(value)
        else:
            print "Error: variable / parameter "+name+" does not exist in population object."
        
    def generate_cpp_add(self):
        """
        In case of cpp_stand_alone compilation, this function generates
        the connector calls.
        """
        if self.projection.connector != None:
            return ('net_->connect('+
                str(self.projection.pre.id)+', '+
                str(self.projection.post.id)+', '+
                self.projection.connector.cpp_call() +', '+ 
                str(self.proj_class['ID'])+', '+ 
                str(self.projection.post.generator.targets.index(self.projection.target))+
                ');\n')
        else:
            print '\tWARNING: no connector object provided.'
            return ''

    def generate(self, verbose):
        """
        generate projection c++ code.
        """
            
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
        
        def member_def(parsed_variables):
            """
            create variable/parameter header entries.
            """
            code = ''
            for var in parsed_variables:
                if var['name'] in Global._pre_def_synapse:
                    continue
                    
                if var['type'] == 'local':
                    code += """\tstd::vector<%(type)s> %(name)s_;\n""" % { 
                        'type': get_type_name(var['cpp_type']), 
                        'name': var['name']
                    }
                else: 
                    # variable is 
                    #     global (postsynaptic neurons)
                    #     or parameter
                    code += """\t%(type)s %(name)s_;\n""" % { 
                        'type': get_type_name(var['cpp_type']), 
                        'name': var['name']
                    }

            return code

        def init(parsed_variables):
            """
            create variable/parameter constructor entries.
            """
            code = ''            
            for var in parsed_variables:
                if var['name'] == 'psp':
                    continue
                
                if var['cpp_type']==bool:
                    if 'True' in var['init']: # is either True or False
                        code += "\t"+var['name']+"_ = true; \n"
                    else:
                        code += "\t"+var['name']+"_ = false; \n"
                else:
                    code += "\t"+var['init']+"\n"       
                            
            code += '\tdt_ = ' + str(Global.config['dt']) + ';'
            return code

        def destructor(parsed_variables):
            """
            create variable/parameter constructor entries.
            """
            code = ''            

            return code
        
        def init_val(parsed_variables):
            code ="""
            Projection::initValues(rank, value, delay);
            """
            return code
        
        def compute_sum(parsed_variables):
            """
            update the weighted sum code.
            
            TODO: delay mechanism
            """

            #
            # check if 'psp' is contained in variable set
            psp_code = ''
            
            for var in parsed_variables:
                if var['name'] == 'psp':
                    psp_code = var['cpp'].split(' = ')[1]

            if len(psp_code) == 0:
                psp_code = '(*pre_rates_)[rank_[i]] * value_[i];'
            else:
                psp_code = psp_code.replace('value_', 'value_[i]')

            code = """\tsum_ =0.0;
    
    if(delay_.empty() || maxDelay_ == 0)    // no delay
    {
        for(int i=0; i<(int)rank_.size(); i++) {
            sum_ += %(psp)s
        }
    }
    else    // delayed connections
    {
        if(constDelay_) // one delay for all connections
        {
            pre_rates_ = pre_population_->getRates(delay_[0]);
        #ifdef _DEBUG
            std::cout << "pre_rates_: " << (*pre_rates_).size() << "("<< pre_rates_ << "), for delay " << delay_[0] << std::endl;
            for(int i=0; i<(int)(*pre_rates_).size(); i++) {
                std::cout << (*pre_rates_)[i] << " ";
            }
            std::cout << std::endl;
        #endif

            for(int i=0; i<(int)rank_.size(); i++) {
                sum_ += %(psp_const_delay)s
            }
        }
        else    // different delays [0..maxDelay]
        {
            std::vector<DATA_TYPE> delayedRates = pre_population_->getRates(delay_, rank_);

            for(int i=0; i<(int)rank_.size(); i++) {
                sum_ += %(psp_dyn_delay)s
            }
        }
    }
""" % { 'psp': psp_code, 
        'psp_const_delay': psp_code,
        'psp_dyn_delay' : psp_code.replace('(*pre_rates_)', 'delayedRates')
       }
            return code
            
        def local_learn(parsed_variables):
            """
            generate synapse update per pre neuron
            """
            loop = ''
            if self.synapse == None:
                return ''
            
            if self.synapse.order == []:
                for var in parsed_variables:
                    if var['name'] == 'psp':
                        continue
                    if var['type'] == 'global':
                        continue

                    if len(var['cpp']) > 0:
                        loop += '\t\t'+var['cpp']+'\n'
                        
                        if 'min' in var.keys():
                            loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var['name'], 'border': var['min'] }
                        if 'max' in var.keys():
                            loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var['name'], 'border': var['max'] }
                       
            else:
                for var in self.synapse.order:
                    for var2 in parsed_variables:
                        if var == var2['name']:
                            if var2['name'] == 'psp':
                                continue
                            if var2['type'] == 'global':
                                continue

                            if len(var2['cpp']) > 0:
                                loop += '\t\t'+var2['cpp']+'\n'

                                if 'min' in var2.keys():
                                    loop += '''\t\tif (%(name)s_[i] < %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var2['name'], 'border': var2['min'] }
                                if 'max' in var2.keys():
                                    loop += '''\t\tif (%(name)s_[i] > %(border)s) \n\t\t\t%(name)s_[i] = %(border)s;\n''' % { 'name': var2['name'], 'border': var2['max'] }


            code = '\tfor(int i=0; i<(int)rank_.size();i++) {\n'
            code += loop
            code += '\t}\n'
            return code

        def global_learn(parsed_variables):
            """
            generate synapse update per post neuron
            """
            
            code = ''
            if self.synapse == None:
                return code
            
            for var in parsed_variables:
                if var['name'] == 'psp':
                    continue
                if var['type'] == 'local':
                    continue
                
                if len(var['cpp']) > 0:
                    code += '\t\t'+var['cpp']+'\n'
                    
                    if 'min' in var.keys():
                        code += '''\t\tif (%(name)s_ < %(border)s) \n\t\t\t%(name)s_ = %(border)s;\n''' % { 'name': var['name'], 'border': var['min'] }
                    if 'max' in var.keys():
                        code += '''\t\tif (%(name)s_ > %(border)s) \n\t\t\t%(name)s_ = %(border)s;\n''' % { 'name': var['name'], 'border': var['max'] }

            return code

        def generate_accessor(synapse_values):
            """
            Creates for all variables/parameters of the synapse the 
            corresponding set and get methods. 
            """
            access = ''
    
            for value in synapse_values:
                if value['name'] in Global._pre_def_synapse:
                    continue
    
                if value['type'] == 'local':
                    access += """
void set%(Name)s(std::vector<%(type)s> %(name)s) { this->%(name)s_= %(name)s; }

std::vector<%(type)s> get%(Name)s() { return this->%(name)s_; }""" % {
       'Name': value['name'].capitalize(),
       'type': get_type_name(value['cpp_type']),
       'name': value['name']
}              
                else:
                    access += """
void set%(Name)s(%(type)s %(name)s) { this->%(name)s_=%(name)s; }

%(type)s get%(Name)s() { return this->%(name)s_; }
""" % {
       'Name': value['name'].capitalize(),
       'type': get_type_name(value['cpp_type']),
       'name': value['name']
}                    
            return access

        if verbose:
            print "    for", self.name, '( from',self.projection.pre.name,'to',self.projection.post.name, ', target = \"',self.projection.target, '\")' 

        # generate func body            
        self.parser = parser.SynapseAnalyser(self.synapse_variables, self.projection.pre.generator.targets,  self.projection.post.generator.targets)
        self.parsed_variables, self.global_operations = self.parser.parse()

        header = '''#ifndef __%(name)s_H__
#define __%(name)s_H__

#include "Global.h"
#include "Includes.h"

class %(name)s : public Projection {
public:
%(name)s(Population* pre, Population* post, int postRank, int target);

%(name)s(int preID, int postID, int postRank, int target);

~%(name)s();

class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }

void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());

void computeSum();

void globalLearn();

void localLearn();

%(access)s
private:
%(synapseMember)s

%(pre_type)s* pre_population_;
%(post_type)s* post_population_;
};
#endif
''' % { 'name': self.name, 
        'pre_type': self.projection.pre.cpp_class,
        'post_type': self.projection.post.cpp_class,
        'access': generate_accessor(self.parsed_variables),
        'synapseMember': member_def(self.parsed_variables) }

        body = '''#include "%(name)s.h"        
#include "Global.h"
using namespace ANNarchy_Global;
        
%(name)s::%(name)s(Population* pre, Population* post, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);
    
    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(name)s::%(name)s(int preID, int postID, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(name)s::~%(name)s() {
#ifdef _DEBUG
    std::cout<<"%(name)s::Destructor"<<std::endl;
#endif
%(destruct)s
}

void %(name)s::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
%(init_val)s
}

void %(name)s::computeSum() {
   
%(sum)s
}

void %(name)s::localLearn() {
%(local)s
}

void %(name)s::globalLearn() {
%(global)s
}

''' % { 'name': self.name,
        'destruct': destructor(self.parsed_variables),
        'pre_type': self.projection.pre.cpp_class,
        'post_type': self.projection.post.cpp_class,
        'init': init(self.parsed_variables), 
        'init_val': init_val(self.parsed_variables),
        'sum': compute_sum(self.parsed_variables), 
        'local': local_learn(self.parsed_variables), 
        'global': global_learn(self.parsed_variables) }

        with open(self.h_file, mode = 'w') as w_file:
            w_file.write(header)

        with open(self.cpp_file, mode = 'w') as w_file:
            w_file.write(body)
            
        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generate_pyx())            

    def generate_pyx(self):
        """
        Create projection class python extension.
        """
        def get_type_name(type):
            """
            mapping between python types and cython names
            """
            if type==float:
                return 'float'
            elif type==int:
                return 'int'
            elif type==bool:
                return 'bool'
            else:
                print "Unknown type, use default = 'DATA_TYPE'"
                return 'float'
        
        def pyx_func(parsed_synapse):
            """
            function calls to wrap the c++ accessors.
            """
            code = ''
    
            for value in parsed_synapse:
                if value['name'] in Global._pre_def_synapse:
                    continue
   
                if value['type'] == 'local':
                     code += '        vector[float] get'+value['name'].capitalize()+'()\n\n'
                     code += '        void set'+value['name'].capitalize()+'(vector[float] values)\n\n'
                else:
                    code += '        '+get_type_name(value['cpp_type'])+' get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'('+get_type_name(value['cpp_type'])+' value)\n\n'    

            return code
    
        def py_func(parsed_synapse):
            """
            function calls to provide access from python
            to c++ data.
            """            
            code = ''
            
            for value in parsed_synapse:
                if value['name'] in Global._pre_def_synapse:
                    continue
    
                code += '    property '+value['name']+':\n'
                if value['type'] == 'local':
                    #getter
                    code += '        def __get__(self):\n'
                    code += '            return np.array(self.cInhInstance.get'+value['name'].capitalize()+'())\n\n'

                    code += '        def __set__(self, value):\n'
                    code += '            if isinstance(value, np.ndarray)==True:\n'
                    code += '                if value.ndim==1:\n'
                    code += '                    self.cInhInstance.set'+value['name'].capitalize()+'(value)\n'
                    code += '                else:\n'
                    code += '                    self.cInhInstance.set'+value['name'].capitalize()+'(value.reshape(self.size))\n'

                    code += '            else:\n'
                    code += '                self.cInhInstance.set'+value['name'].capitalize()+'(np.ones(self.size)*value)\n\n'    
    
                else:
                    #getter
                    code += '        def __get__(self):\n'
                    code += '            return self.cInhInstance.get'+value['name'].capitalize()+'()\n\n'
                
                    code += '        def __set__(self, value):\n'
                    code += '            self.cInhInstance.set'+value['name'].capitalize()+'(value)\n\n'
    
            return code
        
        pyx = '''from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

import numpy as np

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(int preLayer, int postLayer, int postNeuronRank, int target)

%(cFunction)s

cdef class Local%(name)s(LocalProjection):

    cdef %(name)s* cInhInstance

    def __cinit__(self, proj_type, preID, postID, rank, target):
        self.cInhInstance = <%(name)s*>(createProjInstance().getInstanceOf(proj_type, preID, postID, rank, target))

%(pyFunction)s

''' % { 'name': self.proj_class['name'], 
        'cFunction': pyx_func(self.parsed_variables), 
        'pyFunction': py_func(self.parsed_variables) 
    }
        return pyx
