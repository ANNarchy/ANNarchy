
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

%(stop_condition)s

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

# Template for a local variable
# 
# Depends on:
# 
#     * name : name of the variable
#
#     * type : type of the variable
#
local_variable_access = \
"""
    // Access methods for the local variable %(name)s
    std::vector<%(type)s> get_%(name)s() { return this->%(name)s_; }
    void set_%(name)s(std::vector<%(type)s> %(name)s) { this->%(name)s_ = %(name)s; }
    
    %(type)s get_single_%(name)s(int rank) { return this->%(name)s_[rank]; }
    void set_single_%(name)s(int rank, %(type)s %(name)s) { this->%(name)s_[rank] = %(name)s; }

    std::vector< std::vector< %(type)s > > get_recorded_%(name)s() { return this->recorded_%(name)s_; }                    
    void start_record_%(name)s() { this->record_%(name)s_ = true; }
    void stop_record_%(name)s() { this->record_%(name)s_ = false; }
    void clear_recorded_%(name)s() { this->recorded_%(name)s_.clear(); }
"""

# Template for a global variable
# 
# Depends on:
# 
#     * name : name of the variable
#
#     * type : type of the variable
#
global_variable_access = \
"""
    // Access methods for the global variable %(name)s
    %(type)s get_%(name)s() { return this->%(name)s_; }
    void set_%(name)s(%(type)s %(name)s) { this->%(name)s_ = %(name)s; }
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

%(stop_condition)s

%(single_global_ops)s
"""

rate_prepare_neurons="""
    if (maxDelay_ > dt_)
    {
    #ifdef _DEBUG_DELAY
        std::cout << name_ << ": delay = " << maxDelay_ << std::endl;
        std::cout << "OLD ( t = "<< ANNarchy_Global::time << ")" << std::endl;
        for ( int i = 0; i < delayedRates_.size(); i++)
        {
            std::cout << i <<": ";
            for ( auto it = delayedRates_[i].begin(); it != delayedRates_[i].end(); it++)
                std::cout << *it << " ";
            std::cout << std::endl;            
        }
    #endif
    
        delayedRates_.push_front(r_);
        delayedRates_.pop_back();
        
    #ifdef _DEBUG_DELAY
        std::cout << "NEW ( t = "<< ANNarchy_Global::time << ")" << std::endl;
        for ( int i = 0; i < delayedRates_.size(); i++)
        {
            std::cout << i <<": ";
            for ( auto it = delayedRates_[i].begin(); it != delayedRates_[i].end(); it++)
                std::cout << *it << " ";
            std::cout << std::endl;            
        }
    #endif
    }
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
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();
    
    void reset();    // called by global_operations

    void reset(int rank);    // called by metaStep during refractoring phase

%(stop_condition)s

%(global_ops_access)s
    
%(access)s

%(functions)s

private:
    
%(global_ops_method)s

%(member)s

%(random)s
    
    %(friend)s
};
#endif
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

%(stop_condition)s

void %(class)s::reset() 
{
    if (!reset_.empty())
    {
        for (auto it = reset_.begin(); it != reset_.end(); it++)
        {
%(reset_event)s

            refractory_counter_[*it] = refractory_times_[*it];
        }
        
        reset_.erase(reset_.begin(), reset_.end());
    }
    
}

void %(class)s::reset(int rank)
{
    %(reset_neuron)s
}

%(single_global_ops)s
"""

spike_emission_template = """
    if( %(cond)s )
    {
        if (refractory_counter_[i] < 1)
        {
            #pragma omp critical
            {
                //std::cout << "emit spike (pop " << name_ <<")["<<i<<"] ( time="<< ANNarchy_Global::time<< ")" << std::endl;
                this->propagate_.push_back(i);
                this->reset_.push_back(i);
                
                lastSpike_[i] = ANNarchy_Global::time;
                if(record_spike_){
                    spike_timings_[i].push_back(ANNarchy_Global::time);
                }
                spiked_[i] = true;
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
        
        vector[ vector[int] ] get_spike_timings()        
        void reset_spike_timings()
        void start_record_spike()
        void stop_record_spike()
        
        void resetToInit()
        
        void setMaxDelay(int)
        
        void setRefractoryTimes(vector[int])
        
        vector[int] getRefractoryTimes()


%(cFunction)s


cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size):
        self.cInstance = new %(class_name)s('%(name)s', size)

    def name(self):
        return self.cInstance.getName()

    cpdef np.ndarray _get_recorded_spike(self):
        cdef np.ndarray tmp
        tmp = np.array( self.cInstance.get_spike_timings() )
        self.cInstance.reset_spike_timings()
        return tmp

    def _start_record_spike(self):
        self.cInstance.start_record_spike()

    def _stop_record_spike(self):
        self.cInstance.stop_record_spike()

    def reset(self):
        self.cInstance.resetToInit()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "py%(name)s.size is a read-only attribute."

    cpdef np.ndarray _get_refractory_times(self):
        return np.array(self.cInstance.getRefractoryTimes())
        
    cpdef _set_refractory_times(self, np.ndarray value):
        self.cInstance.setRefractoryTimes(value)
            
%(pyFunction)s
"""

# Local Cython property
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * type : The type of the variable
local_property_pyx = """

    # local: %(name)s
    cpdef np.ndarray _get_%(name)s(self):
        return np.array(self.cInstance.get_%(name)s())
        
    cpdef _set_%(name)s(self, np.ndarray value):
        self.cInstance.set_%(name)s(value)
        
    cpdef %(type)s _get_single_%(name)s(self, rank):
        return self.cInstance.get_single_%(name)s(rank)

    def _set_single_%(name)s(self, int rank, %(type)s value):
        self.cInstance.set_single_%(name)s(rank, value)

    def _start_record_%(name)s(self):
        self.cInstance.start_record_%(name)s()

    def _stop_record_%(name)s(self):
        self.cInstance.stop_record_%(name)s()

    cpdef np.ndarray _get_recorded_%(name)s(self):
        tmp = np.array(self.cInstance.get_recorded_%(name)s())
        self.cInstance.clear_recorded_%(name)s()
        return tmp
        
"""

# Global Cython property
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * type : The type of the variable
global_property_pyx = """

    # global: %(name)s
    cpdef %(type)s _get_%(name)s(self):
        return self.cInstance.get_%(name)s()

    cpdef _set_%(name)s(self, %(type)s value):
        self.cInstance.set_%(name)s(value)
        
"""

# Local Cython wrapper
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * type : C type of variable
local_wrapper_pyx = """
        # Local %(name)s
        vector[%(type)s] get_%(name)s()
        void set_%(name)s(vector[%(type)s] values)
        %(type)s get_single_%(name)s(int rank)
        void set_single_%(name)s(int rank, %(type)s values)
        void start_record_%(name)s()
        void stop_record_%(name)s()
        void clear_recorded_%(name)s()
        vector[vector[%(type)s]] get_recorded_%(name)s()
"""

# Global Cython wrapper
# 
# Depends on:
# 
#     * name : name of the variable
#    
#     * type : C type of variable
global_wrapper_pyx = """
        # Global %(name)s
        %(type)s get_%(name)s()
        void set_%(name)s(%(type)s value)                
"""
