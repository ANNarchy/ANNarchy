
# Header for a Rate population.
# 
# Depends on:
# 
#     * class : the class name (e.g. Population1)
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
rate_population_header = \
"""#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

%(global_ops_access)s
    
%(access)s

private:

%(member)s

%(global_ops_method)s

%(random)s

};
#endif
"""

# Header for a Spike population.
# 
# Depends on:
# 
#     * class : the class name (e.g. Population1)
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
spike_population_header = \
"""#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

    void propagateSpike();
    
    void reset();

%(global_ops_access)s
    
%(access)s

private:

%(member)s

%(global_ops_method)s

%(random)s

    std::vector<int> propagate_;    ///< neurons which will propagate their spike
    std::vector<int> reset_;    ///< neurons which will reset after current eval
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

    %(type)s getSingle%(Name)s(int rank) { return this->%(name)s_[rank]; }
    void setSingle%(Name)s(int rank, %(type)s %(name)s) { this->%(name)s_[rank] = %(name)s; }

    std::vector< std::vector< %(type)s > >getRecorded%(Name)s() { return this->recorded_%(name)s_; }                    
    void startRecord%(Name)s() { this->record_%(name)s_ = true; }
    void stopRecord%(Name)s() { this->record_%(name)s_ = false; }
    void clearRecorded%(Name)s() { this->recorded_%(name)s_.clear(); }
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

# Body for a rate population
#
# Depends on:
#
#    * class : the class name
#
#    * constructor : code for the constructor where all variables are initialized
#
#    * destructor : code for the destructor where all variables are freed
# 
#    * resetToInit : code for the reinitialization
# 
#    * metaStep : code for the metastep function
# 
#    * global_ops : code for computing the global operations
# 
#    * record : code for the recording
#
#    * single_global_ops : code for the single global operations
rate_population_body = """#include "%(class)s.h"
#include "Global.h"
using namespace ANNarchy_Global;

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(constructor)s
    Network::instance()->addPopulation(this);
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif
%(destructor)s
}

void %(class)s::resetToInit() {
%(resetToInit)s
}

void %(class)s::localMetaStep(int i) {
%(localMetaStep)s
}

void %(class)s::globalMetaStep() {
%(globalMetaStep)s        
}

void %(class)s::globalOperations() {
%(global_ops)s
}

void %(class)s::record() {
%(record)s
}

%(single_global_ops)s
"""

# Body for a Spike population
#
# Depends on:
#
#    * class : the class name
#
#    * constructor : code for the constructor where all variables are initialized
#
#    * destructor : code for the destructor where all variables are freed
# 
#    * resetToInit : code for the reinitialization
# 
#    * metaStep : code for the metastep function
# 
#    * global_ops : code for computing the global operations
# 
#    * record : code for the recording
#
#    * single_global_ops : code for the single global operations
spike_population_body = """#include "%(class)s.h"
#include "Global.h"
using namespace ANNarchy_Global;

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(constructor)s

    std::vector<bool>(nbNeurons_, false);
    
    Network::instance()->addPopulation(this);
}

%(class)s::~%(class)s() {
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif
%(destructor)s
}

void %(class)s::resetToInit() {
%(resetToInit)s
}

void %(class)s::localMetaStep(int i) {
%(localMetaStep)s    
}

void %(class)s::globalMetaStep() {
    spiked_ = std::vector<bool>(nbNeurons_, false);

%(globalMetaStep)s    
}

void %(class)s::globalOperations() {
    
    propagateSpike();
    
    reset();

%(global_ops)s
}

void %(class)s::record() {
%(record)s
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
        for (auto it = reset_.begin(); it != reset_.end(); it++)
        {
%(reset_event)s
        }
        
        reset_.erase(reset_.begin(), reset_.end());
    }
    
}

%(single_global_ops)s
"""

# Cython file for a rate population
#
# Depends on:
#
#    * name : the class name
#
#    * cFunction : 
#
#    * neuron_count
# 
#    * pyFunction
# 
rate_population_pyx = """from libcpp cimport bool
from libcpp.vector cimport vector
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
"""

# Cython file for a Spike population
#
# Depends on:
#
#    * name : the class name
#
#    * cFunction : 
#
#    * neuron_count
# 
#    * pyFunction
# 
spike_population_pyx = """from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        vector[ vector[int] ] getSpikeTimings()
        
        void resetToInit()
        
        void setMaxDelay(int)

%(cFunction)s


cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self):
        self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

    def name(self):
        return self.cInstance.getName()

    def get_spike_timings(self):
        return np.array( self.cInstance.getSpikeTimings() )

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
            return np.array(self.cInstance.get%(Name)s())

        def __set__(self, value):
            if isinstance(value, np.ndarray)==True:
                if value.ndim==1:
                    self.cInstance.set%(Name)s(value)
                else:
                    self.cInstance.set%(Name)s(value.reshape(self.size))
            else:
                self.cInstance.set%(Name)s(np.ones(self.size)*value)

    def _get_single_%(name)s(self, rank):
        return self.cInstance.getSingle%(Name)s(rank)

    def _set_single_%(name)s(self, rank, value):
        self.cInstance.setSingle%(Name)s(rank, value)

    def _start_record_%(name)s(self):
        self.cInstance.startRecord%(Name)s()

    def _stop_record_%(name)s(self):
        self.cInstance.stopRecord%(Name)s()

    def _get_recorded_%(name)s(self):
        tmp = np.array(self.cInstance.getRecorded%(Name)s())
        self.cInstance.clearRecorded%(Name)s()
        return tmp
        
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
            return self.cInstance.get%(Name)s()

        def __set__(self, value):
            self.cInstance.set%(Name)s(value)
        
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
        %(type)s getSingle%(Name)s(int rank)
        void setSingle%(Name)s(int rank, %(type)s values)
        void startRecord%(Name)s()
        void stopRecord%(Name)s()
        void clearRecorded%(Name)s()
        vector[vector[%(type)s]] getRecorded%(Name)s()
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
