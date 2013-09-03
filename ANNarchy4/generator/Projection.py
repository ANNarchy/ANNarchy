"""
Projection generator.
"""
from ANNarchy4.core import Global
from ANNarchy4 import parser

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
            self.parsed_variables = synapse.parsed_variables
        else:
            self.parsed_variables = []
            
        self.projection = projection
        self.name = projection.name

        self.h_file = Global.annarchy_dir+'/build/'+self.name+'.h'
        self.cpp_file = Global.annarchy_dir+'/build/'+self.name+'.cpp'
        self.pyx = Global.annarchy_dir+'/pyx/'+self.name+'.pyx'

        self.proj_class = { 
            'ID': len(Global._projections), 
            'name': self.name 
        }
        
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

    def generate(self):
        """
        generate projection c++ code.
        """
        def member_def(parsed_variables):
            """
            create variable/parameter header entries.
            """
            code = ''
            for var in parsed_variables:
                if var['name'] in Global._pre_def_synapse:
                    continue
                    
                if var['type'] == 'parameter':
                    code += "\tDATA_TYPE "+var['name']+"_;\n"
                elif var['type'] == 'local':
                    code += "\tstd::vector<DATA_TYPE> "+var['name']+"_;\n"
                else: # global (postsynaptic neurons), or weight bound
                    code += "\tDATA_TYPE "+var['name']+"_;\n"

            return code

        def init(parsed_variables):
            """
            create variable/parameter constructor entries.
            """
            code = ''
            
           
            for var in parsed_variables:
                if var['name'] == 'psp':
                    continue
                        
                code += "\t"+var['init']+"\n"

            code += '\tdt_ = ' + str(Global.config['dt']) + ';'
            return code
        
        def init_val(parsed_variables):
            code ="""
            Projection::initValues(rank, value, delay);
            
            pre_population_->setMaxDelay(maxDelay_);
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
                    psp_code = var['cpp'].split('=')[1]
                    
            if len(psp_code) == 0:
                psp_code = '(*pre_rates_)[rank_[i]] * value_[i];'

            code = """\tsum_ =0.0;
    
    if(delay_.empty())    // no delay
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
                    if var == 'psp':
                        continue
                    if var == 'global':
                        continue

                    for var2 in parsed_variables:
                        if var == var2['name']:
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
    
                if value['type'] == 'parameter':
                    access += 'void set'+value['name'].capitalize()+'(DATA_TYPE '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += 'DATA_TYPE get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
                elif value['type'] == 'global':
                    access += 'void set'+value['name'].capitalize()+'(DATA_TYPE '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += 'DATA_TYPE get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
                else:
                    access += 'void set'+value['name'].capitalize()+'(std::vector<DATA_TYPE> '+value['name']+') { this->'+value['name']+'_='+value['name']+'; }\n\n'
                    access += 'std::vector<DATA_TYPE> get'+value['name'].capitalize()+'() { return this->'+value['name']+'_; }\n\n'
                    
            return access

        # generate func body            

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
%(name)s::%(name)s(Population* pre, Population* post, int postRank, int target) : Projection() {

    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

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
        
        def pyx_func(parsed_synapse):
            """
            function calls to wrap the c++ accessors.
            """
            code = ''
    
            for value in parsed_synapse:
                if value['name'] in Global._pre_def_synapse:
                    continue
   
                if value['type'] == 'parameter':
                    code += '        float get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'(float value)\n\n'
                else:
                    code += '        vector[float] get'+value['name'].capitalize()+'()\n\n'
                    code += '        void set'+value['name'].capitalize()+'(vector[float] values)\n\n'
    
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
