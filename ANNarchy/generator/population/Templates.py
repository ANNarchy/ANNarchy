
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
#    
#     * functions : inline definition of custom functions    
rate_population_header = \
"""#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"
#include "RatePopulation.h"
using namespace ANNarchy_Global;

class %(class)s: public RatePopulation
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    void prepareNeurons();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

%(global_ops_access)s
    
%(access)s

%(functions)s

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
#    
#     * functions : inline definition of custom functions       
spike_population_header = \
"""#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"
#include "SpikePopulation.h"
using namespace ANNarchy_Global;

class %(class)s: public SpikePopulation
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void prepareNeurons();
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

    void propagateSpike();
    
    void reset();

%(global_ops_access)s
    
%(access)s

%(functions)s

private:

%(member)s

%(global_ops_method)s

%(random)s

    std::vector<int> propagate_;    ///< neurons which will propagate their spike
    std::vector<int> reset_;    ///< neurons which will reset after current eval
    
    %(friend)s
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

%(class)s::%(class)s(std::string name, int nbNeurons): RatePopulation(name, nbNeurons)
{
    rank_ = %(pop_id)s;
    
#ifdef _DEBUG
    std::cout << name << ": %(class)s::%(class)s called (using rank " << rank_ << ")" << std::endl;
#endif

%(constructor)s

    try
    {
        Network::instance()->addPopulation(this);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to attach population"<< std::endl;
        std::cout << e.what() << std::endl;
    }
}

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif
%(destructor)s
}

void %(class)s::prepareNeurons() 
{
%(prepare)s
}

void %(class)s::resetToInit() 
{
%(resetToInit)s
}

void %(class)s::localMetaStep(int i) 
{
%(localMetaStep)s
}

void %(class)s::globalMetaStep() 
{
%(globalMetaStep)s        
}

void %(class)s::globalOperations() 
{
%(global_ops)s
}

void %(class)s::record() 
{
%(record)s
    for(unsigned int p=0; p< projections_.size(); p++)
    {
        projections_[p]->record();
    }
}

%(single_global_ops)s
"""

rate_prepare_neurons="""
    if (maxDelay_ > 1)
    {
    #ifdef _DEBUG
        std::cout << name_ << ": got delayed rates = " << maxDelay_ << std::endl;
    #endif
    
        delayedRates_.push_front(rate_);
        delayedRates_.pop_back();
    }
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
#include "SpikeDendrite.h"
#include "SpikeProjection.h"

%(class)s::%(class)s(std::string name, int nbNeurons): SpikePopulation(name, nbNeurons)
{
    rank_ = %(pop_id)s;
    
#ifdef _DEBUG
    std::cout << name << ": %(class)s::%(class)s called (using rank " << rank_ << ")" << std::endl;
#endif

%(constructor)s

    spiked_ = std::vector<bool>(nbNeurons_, false);
    
    try
    {
        Network::instance()->addPopulation(this);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to attach population"<< std::endl;
        std::cout << e.what() << std::endl;
    }
}

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif
%(destructor)s
}

void %(class)s::prepareNeurons() 
{
%(prepare)s
}

void %(class)s::resetToInit() 
{
%(resetToInit)s
}

void %(class)s::localMetaStep(int i) 
{
%(localMetaStep)s    
}

void %(class)s::globalMetaStep() 
{
    spiked_ = std::vector<bool>(nbNeurons_, false);

%(globalMetaStep)s    
}

void %(class)s::globalOperations() 
{
    reset();
    
    propagateSpike();

%(global_ops)s
}

void %(class)s::record() 
{
%(record)s
    for(unsigned int p=0; p< projections_.size(); p++)
    {
        projections_[p]->record();
    }
}

void %(class)s::propagateSpike() {

    if (!propagate_.empty())
    {
        for(auto n_it= propagate_.begin(); n_it!= propagate_.end(); n_it++)
        {
            // emit a postsynaptic spike on outgoing projections
            for( auto p_it = spikeTargets_[(*n_it)].begin(); p_it != spikeTargets_[(*n_it)].end(); p_it++)
            {
                static_cast<SpikeDendrite*>(*p_it)->preEvent(*n_it);
            }
            
        }
        
        // emit a postsynaptic spike on receiving projections
        for( auto p_it = projections_.begin(); p_it != projections_.end(); p_it++)
        {
            static_cast<SpikeProjection*>(*p_it)->postEvent(propagate_);
            
        }
        
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
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        void resetToInit()
        
        void setMaxDelay(int)

%(cFunction)s


cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size):
        self.cInstance = new %(class_name)s('%(name)s', size)

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
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N)

        int getNeuronCount()
        
        string getName()
        
        vector[ vector[int] ] getSpikeTimings()
        
        void resetSpikeTimings()
        
        void resetToInit()
        
        void setMaxDelay(int)

%(cFunction)s


cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size):
        self.cInstance = new %(class_name)s('%(name)s', size)

    def name(self):
        return self.cInstance.getName()

    cpdef list get_spike_timings(self):
        return list( self.cInstance.getSpikeTimings() )

    def reset_spike_timings(self):
        self.cInstance.resetSpikeTimings() 

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
#    
#     * type : The type of the variable
local_property_pyx = """

    # local: %(name)s
    cpdef np.ndarray _get_%(name)s(self):
        return np.array(self.cInstance.get%(Name)s())
        
    cpdef _set_%(name)s(self, np.ndarray value):
        self.cInstance.set%(Name)s(value)
        
    cpdef %(type)s _get_single_%(name)s(self, rank):
        return self.cInstance.getSingle%(Name)s(rank)

    def _set_single_%(name)s(self, int rank, %(type)s value):
        self.cInstance.setSingle%(Name)s(rank, value)

    def _start_record_%(name)s(self):
        self.cInstance.startRecord%(Name)s()

    def _stop_record_%(name)s(self):
        self.cInstance.stopRecord%(Name)s()

    cpdef np.ndarray _get_recorded_%(name)s(self):
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

    # global: %(name)s
    cpdef %(type)s _get_%(name)s(self):
        return self.cInstance.get%(Name)s()

    cpdef _set_%(name)s(self, %(type)s value):
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
