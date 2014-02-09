
# Header for a Rate projection.
# 
# Depends on:
# 
#     * name : the class name (e.g. Projection1)
#    
#     * access : public access methods for all parameters and variables
#    
#     * global_ops_access : access to the global operations (min, max, mean, etc)
#    
#     * global_ops_method : methods for the global operations (min, max, mean, etc)
#    
#     * member : private definition of parameters and variables    
#    
#     * random : private definition of RandomDistribution arrays    
rate_projection_header = \
"""#ifndef __%(class)s_H__
#define __%(class)s_H__

#include "Global.h"
#include "Includes.h"

class %(class)s : public Projection {
public:
    %(class)s(Population* pre, Population* post, int postRank, int target);
    
    %(class)s(int preID, int postID, int postRank, int target);
    
    ~%(class)s();
    
    class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());
    
    void computeSum();
    
    void globalLearn();
    
    void localLearn();

%(access)s
private:
%(member)s

    %(pre_name)s* pre_population_;
    %(post_name)s* post_population_;
};
#endif
""" 

# Header for a spike projection.
# 
# Depends on:
# 
#     * name : the class name (e.g. Projection1)
#    
#     * access : public access methods for all parameters and variables
#    
#     * global_ops_access : access to the global operations (min, max, mean, etc)
#    
#     * global_ops_method : methods for the global operations (min, max, mean, etc)
#    
#     * member : private definition of parameters and variables    
#    
#     * random : private definition of RandomDistribution arrays    
spike_projection_header = \
"""#ifndef __%(class)s_H__
#define __%(class)s_H__

#include "Global.h"
#include "Includes.h"

class %(class)s : public Projection {
public:
    %(class)s(Population* pre, Population* post, int postRank, int target);
    
    %(class)s(int preID, int postID, int postRank, int target);
    
    ~%(class)s();
    
    class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());
    
    void computeSum();
    
    void globalLearn();
    
    void localLearn();

    void propagateSpike();
    
%(access)s
private:
%(member)s

    %(pre_name)s* pre_population_;
    %(post_name)s* post_population_;
};
#endif
"""

# Template for a local variable
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
#
#     * type : type of the variable
#
local_variable_access = \
"""
    // Access methods for the local variable %(name)s
    std::vector<%(type)s> get%(Name)s() { return this->%(name)s_; }
    void set%(Name)s(std::vector<%(type)s> %(name)s) { this->%(name)s_ = %(name)s; }
"""

# Template for a global variable
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
#
#     * type : type of the variable
#
global_variable_access = \
"""
    // Access methods for the global variable %(name)s
    %(type)s get%(Name)s() { return this->%(name)s_; }
    void set%(Name)s(%(type)s %(name)s) { this->%(name)s_ = %(name)s; }
"""

# Body for a rate projection
#
# Depends on:
#
#    * class : the class name
#
#    * destructor : code for the destructor where all variables are freed
# 
#    * pre_type : name of class of the presynaptic population
# 
#    * post_type : name of class of the presynaptic population
# 
#    * init : initial values for parameters and variables
# 
#    * init_val : code for rank, values, delays
# 
#    * local : code for local_learn 
#
#    * global : code for global_learn
#
rate_projection_body = \
"""#include "%(class)s.h"        
#include "Global.h"
using namespace ANNarchy_Global;
        
%(class)s::%(class)s(Population* pre, Population* post, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);
    
    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(class)s::%(class)s(int preID, int postID, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout<<"%(class)s::Destructor"<<std::endl;
#endif

%(destructor)s
}

void %(class)s::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
    Projection::initValues(rank, value, delay);
%(init_val)s
}

void %(class)s::computeSum() {   
%(sum)s
}

void %(class)s::localLearn() {
%(local)s
}

void %(class)s::globalLearn() {
%(global)s
}

"""

# Body for a Spike projection
#
# Depends on:
#
#    * class : the class name
#
#    * destructor : code for the destructor where all variables are freed
# 
#    * pre_type : name of class of the presynaptic population
# 
#    * post_type : name of class of the presynaptic population
# 
#    * init : initial values for parameters and variables
# 
#    * init_val : code for rank, values, delays
# 
#    * local : code for local_learn 
#
#    * global : code for global_learn
#
spike_projection_body = \
"""#include "%(class)s.h"        
#include "Global.h"
using namespace ANNarchy_Global;
        
%(class)s::%(class)s(Population* pre, Population* post, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);
    
    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(class)s::%(class)s(int preID, int postID, int postRank, int target) : Projection() {
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addProjection(postRank, this);
    
%(init)s
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout<<"%(class)s::Destructor"<<std::endl;
#endif

%(destructor)s
}

void %(class)s::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
    Projection::initValues(rank, value, delay);
%(init_val)s
}

void %(class)s::computeSum() {   
%(sum)s
}

void %(class)s::localLearn() {
%(local)s
}

void %(class)s::globalLearn() {
%(global)s
}

void %(class)s::propagateSpike() 
{
    auto s_it = value_.begin();
    auto n_it = rank_.begin();
    for(n_it, s_it; n_it != rank_.end(); n_it++,s_it++)
    {
        pre_population_->inc_g_%(target)s((*n_it), (*s_it));
    }
}

"""

# Template for the computeSum() method of a projection
#
# * psp: basic code for the psp (default (*pre_rates_)[rank_[i]] * value_[i];) 
#
# * psp_const_delay : code when the delay is constant (normally the same as psp)
#
# * psp_dyn_delay : code when delays are variable (default delayedRates[rank_[i]] * value_[i];) 
#
psp_code_body = \
"""
    sum_ =0.0;
    
    if(delay_.empty() || maxDelay_ == 0)    // no delay
    {
                    
        for(int i=0; i<(int)rank_.size(); i++) 
        {
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
            #pragma omp for schedule(static)
            for(int i=0; i<(int)rank_.size(); i++) {
                sum_ += %(psp_const_delay)s
            }
        }
        else    // different delays [0..maxDelay]
        {
            std::vector<DATA_TYPE> delayedRates = pre_population_->getRates(delay_, rank_);

            #pragma omp for schedule(static)
            for(int i=0; i<(int)rank_.size(); i++) {
                sum_ += %(psp_dyn_delay)s
            }
        }
    }
""" 

# Cython file for a rate population
#
# Depends on:
#
#    * name : the class name
#
#    * cFunction : c++ methods to access attributes
# 
#    * pyFunction ' python functions to access attributes
# 
rate_projection_pyx = \
"""from libcpp.vector cimport vector
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

""" 

# Cython file for a Spike projection
#
# Depends on:
#
#    * name : the class name
#
#    * cFunction : c++ methods to access attributes
# 
#    * pyFunction ' python functions to access attributes
# 
spike_projection_pyx = \
"""from libcpp.vector cimport vector
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

""" 

# Local Cython property
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
local_property_pyx = """
    property %(name)s:
        def __get__(self):
            return np.array(self.cInhInstance.get%(Name)s())

        def __set__(self, value):
            if isinstance(value, np.ndarray)==True:
                if value.ndim==1:
                    self.cInhInstance.set%(Name)s(value)
                else:
                    self.cInhInstance.set%(Name)s(value.reshape(self.size))
            else:
                self.cInhInstance.set%(Name)s(np.ones(self.size)*value)

#    def _get_single_%(name)s(self, rank):
#        return self.cInhInstance.getSingle%(Name)s(rank)
#
#    def _set_single_%(name)s(self, rank, value):
#        self.cInhInstance.setSingle%(Name)s(rank, value)
#
#    def _start_record_%(name)s(self):
#        self.cInhInstance.startRecord%(Name)s()

#    def _stop_record_%(name)s(self):
#        self.cInhInstance.stopRecord%(Name)s()

#    def _get_recorded_%(name)s(self):
#        tmp = np.array(self.cInhInstance.getRecorded%(Name)s())
#        self.cInhInstance.clearRecorded%(Name)s()
#        return tmp
        
"""

# Global Cython property
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
global_property_pyx = """
    property %(name)s:
        def __get__(self):
            return self.cInhInstance.get%(Name)s()

        def __set__(self, value):
            self.cInhInstance.set%(Name)s(value)
        
"""

# Local Cython wrapper
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
#    
#     * type : C type of variable
local_wrapper_pyx = """
        # Local %(name)s
        vector[%(type)s] get%(Name)s()
        void set%(Name)s(vector[%(type)s] values)
        #%(type)s getSingle%(Name)s(int rank)
        #void setSingle%(Name)s(int rank, %(type)s values)
        #void startRecord%(Name)s()
        #void stopRecord%(Name)s()
        #void clearRecorded%(Name)s()
        #vector[vector[%(type)s]] getRecorded%(Name)s()
"""

# Global Cython wrapper
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * Name : Capitalized name of variable
#    
#     * type : C type of variable
global_wrapper_pyx = """
        # Global %(name)s
        %(type)s get%(Name)s()
        void set%(Name)s(%(type)s value)                
"""