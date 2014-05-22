# Header for a rate dendrite.
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
#    
#     * functions : inline definition of custom functions     
rate_projection_header = \
"""#ifndef __%(class)s_H__
#define __%(class)s_H__

#include "Global.h"
#include "Includes.h"
#include "RateProjection.h"

class %(class)s : public RateProjection 
{
public:
    %(class)s(Population* pre, Population* post, int target);
    
    %(class)s(int preID, int postID, int target);
    
    ~%(class)s();

    std::vector<int> getRank(int post_rank);
    
    std::vector<DATA_TYPE> getValue(int post_rank);    
    void setValue(int post_rank, std::vector<DATA_TYPE> values);
    
    std::vector<int> getDelay(int post_rank);
    
    Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    void addDendrite(int, std::vector<int>, std::vector<DATA_TYPE>, std::vector<int>);
        
%(access)s

private:

    %(pre_name)s* pre_population_;
    %(post_name)s* post_population_;
};
#endif
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
#    * local : code for local_learn 
#
#    * global : code for global_learn
#
rate_projection_body = \
"""#include "%(class)s.h"        
#include "Global.h"
#include "%(dend_class)s.h"

%(add_include)s

using namespace ANNarchy_Global;

%(class)s::%(class)s(Population* pre, Population* post, int target) : RateProjection() 
{
#ifdef _DEBUG
    std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << "', target = '" << target << "', coding = 'rate' ) " << std::endl;
#endif
    pre_population_ = static_cast<class %(pre_type)s*>( pre );
    post_population_ = static_cast<class %(post_type)s*>( post );

    target_ = target;
    post_population_->addProjection(this);

    nbDendrites_ = static_cast<int>(post_population_->getNeuronCount());
    dendrites_ = std::vector< Dendrite* >(nbDendrites_, NULL);
}

%(class)s::%(class)s(int pre, int post, int target) : RateProjection() 
{
#ifdef _DEBUG
    std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << "', target = '" << target << "', coding = 'rate' ) " << std::endl;
#endif
    pre_population_ = static_cast<class %(pre_type)s*>(Network::instance()->getPopulation(pre));
    post_population_ = static_cast<class %(post_type)s*>(Network::instance()->getPopulation(post));

    target_ = target;
    post_population_->addProjection(this);

    nbDendrites_ = static_cast<int>(post_population_->getNeuronCount());
    dendrites_ = std::vector< Dendrite* >(nbDendrites_, NULL);
}

std::vector<int> %(class)s::getRank(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getRank();
}

std::vector<DATA_TYPE> %(class)s::getValue(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getValue();
}

void %(class)s::setValue(int post_rank, std::vector<DATA_TYPE> values)
{
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->setValue(values);
}

std::vector<int> %(class)s::getDelay(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getDelay();
}

%(access)s

void %(class)s::addDendrite(int post_rank, std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay)
{
    dendrites_[post_rank] = static_cast<RateDendrite*>(new %(dend_class)s(pre_population_, post_population_, post_rank, target_));
    
    dendrites_[post_rank]->setRank(rank);
    dendrites_[post_rank]->setValue(value);
    dendrites_[post_rank]->setDelay(delay);    
}
"""

# Header for a spike dendrite.
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
#    
#     * functions : inline definition of custom functions      
spike_projection_header = \
"""#ifndef __%(class)s_H__
#define __%(class)s_H__

#include "Global.h"
#include "Includes.h"
#include "SpikeProjection.h"

class %(class)s : public SpikeProjection 
{
public:
    %(class)s(Population* pre, Population* post, int target);
    
    %(class)s(int preID, int postID, int target);
    
    ~%(class)s();

    std::vector<int> getRank(int post_rank);
    
    std::vector<DATA_TYPE> getValue(int post_rank);
    void setValue(int post_rank, std::vector<DATA_TYPE> values);
    
    std::vector<int> getDelay(int post_rank);
    
    Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    void addDendrite(int, std::vector<int>, std::vector<DATA_TYPE>, std::vector<int>);
    
%(access)s

protected:

    %(pre_name)s* pre_population_;
    %(post_name)s* post_population_;
};
#endif
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
#    * local : code for local_learn 
#
#    * global : code for global_learn
#
spike_projection_body = \
"""#include "%(class)s.h"        
#include "Global.h"
#include "%(dend_class)s.h"

%(add_include)s

using namespace ANNarchy_Global;

%(class)s::%(class)s(Population* pre, Population* post, int target) : SpikeProjection() 
{
#ifdef _DEBUG
    std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << "', target = '" << target << "', coding = 'rate' ) " << std::endl;
#endif
    pre_population_ = static_cast<class %(pre_type)s*>( pre );
    post_population_ = static_cast<class %(post_type)s*>( post );

    target_ = target;
    post_population_->addProjection(this);

    nbDendrites_ = static_cast<int>(post_population_->getNeuronCount());
    dendrites_ = std::vector< Dendrite* >(nbDendrites_, NULL);
}

%(class)s::%(class)s(int pre, int post, int target) : SpikeProjection() 
{
#ifdef _DEBUG
    std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << "', target = '" << target << "', coding = 'rate' ) " << std::endl;
#endif
    pre_population_ = static_cast<class %(pre_type)s*>(Network::instance()->getPopulation(pre));
    post_population_ = static_cast<class %(post_type)s*>(Network::instance()->getPopulation(post));

    target_ = target;
    post_population_->addProjection(this);

    nbDendrites_ = static_cast<int>(post_population_->getNeuronCount());
    dendrites_ = std::vector< Dendrite* >(nbDendrites_, NULL);
}

std::vector<int> %(class)s::getRank(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getRank();
}

std::vector<DATA_TYPE> %(class)s::getValue(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getValue();
}

void %(class)s::setValue(int post_rank, std::vector<DATA_TYPE> values)
{
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->setValue(values);
}

std::vector<int> %(class)s::getDelay(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getDelay();
}

%(access)s

void %(class)s::addDendrite(int post_rank, std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay)
{
    dendrites_[post_rank] = static_cast<SpikeDendrite*>(new %(dend_class)s(pre_population_, post_population_, post_rank, target_));
    
    dendrites_[post_rank]->setRank(rank);
    dendrites_[post_rank]->setValue(value);
    dendrites_[post_rank]->setDelay(delay);    
}
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
local_idx_variable_access = \
"""
    // Access methods for the local variable %(name)s
    std::vector<%(type)s> get%(Name)s(int post_rank);
    void set%(Name)s(int post_rank, std::vector<%(type)s> %(name)s);
    
    %(type)s getSingle%(Name)s(int post_rank, int rank);
    void setSingle%(Name)s(int post_rank, int rank, %(type)s %(name)s);

    std::vector< std::vector< %(type)s > >getRecorded%(Name)s(int post_rank);                    
    void startRecord%(Name)s(int post_rank);
    void stopRecord%(Name)s(int post_rank);
    void clearRecorded%(Name)s(int post_rank);
"""

local_idx_variable_access_body = \
"""
// Access methods for the local variable %(name)s
std::vector<%(type)s> %(class)s::get%(Name)s(int post_rank) 
{     
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get%(Name)s(); 
}

void %(class)s::set%(Name)s(int post_rank, std::vector<%(type)s> %(name)s) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->set%(Name)s(%(name)s); 
}

%(type)s %(class)s::getSingle%(Name)s(int post_rank, int rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getSingle%(Name)s(rank); 
}
void %(class)s::setSingle%(Name)s(int post_rank, int rank, %(type)s %(name)s) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->setSingle%(Name)s(rank, %(name)s); 
}

std::vector< std::vector< %(type)s > > %(class)s::getRecorded%(Name)s(int post_rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->getRecorded%(Name)s(); 
}                    
void %(class)s::startRecord%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->startRecord%(Name)s(); 
}
void %(class)s::stopRecord%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->stopRecord%(Name)s(); 
}
void %(class)s::clearRecorded%(Name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->clearRecorded%(Name)s(); 
}
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
global_idx_variable_access = \
"""
    // Access methods for the global variable %(name)s
    %(type)s get%(Name)s(int post_rank);
    void set%(Name)s(int post_rank, %(type)s %(name)s);
"""

global_idx_variable_access_body = \
"""
    // Access methods for the global variable %(name)s
    %(type)s %(class)s::get%(Name)s(int post_rank) 
    {  
        return (static_cast<class %(dend_class)s*>(dendrites_[post_rank]))->get%(Name)s(); 
    }
    
    void %(class)s::set%(Name)s(int post_rank, %(type)s %(name)s) 
    { 
        (static_cast<class %(dend_class)s*>(dendrites_[post_rank]))->set%(Name)s( %(name)s ); 
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
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector import CSR as CSR

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(int preLayer, int postLayer, int target)

        void addDendrite(int, vector[int], vector[float], vector[int])

        void addSynapse(int, int, float, int)

        void removeSynapse(int, int)

        void initValues(int post_rank)
         
        vector[int] getRank(int post_rank)

        # local variable value
        vector[float] getValue(int post_rank)
        void setValue(int post_rank, vector[float] values)
        void startRecordValue(int post_rank)
        void stopRecordValue(int post_rank)
        void clearRecordedValue(int post_rank)
        vector[vector[float]] getRecordedValue(int post_rank)
        
        vector[int] getDelay(int post_rank)
        
        int nbDendrites()

        int nbSynapses(int post_rank)
        
        void setLearning(bool)
    
        void setLearnFrequency(int)
    
        int getLearnFrequency()
    
        void setLearnOffset( int )
    
        int getLearnOffset( )
    
        bool isLearning()
        
%(cFunction)s

cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self, preID, postID, target):
        self.cInstance = new %(name)s(preID, postID, target)

    cpdef add_synapse(self, int post_rank, int pre_rank, float value, int delay):
        self.cInstance.addSynapse(post_rank, pre_rank, value, delay)

    cpdef remove_synapse(self, int post_rank, int pre_rank):
        self.cInstance.removeSynapse(post_rank, pre_rank)

    # Rank (read only)
    cpdef np.ndarray _get_rank(self, int post_rank):
        return np.array(self.cInstance.getRank(post_rank))

    # Value
    cpdef np.ndarray _get_value(self, int post_rank):
        return np.array(self.cInstance.getValue(post_rank))

    cpdef np.ndarray _set_value(self, int post_rank, np.ndarray value ):
        self.cInstance.setValue(post_rank, value)

    def _start_record_value(self, int post_rank):
        self.cInstance.startRecordValue(post_rank)

    def _stop_record_value(self, int post_rank):
        self.cInstance.stopRecordValue(post_rank)

    cpdef np.ndarray _get_recorded_value(self, int post_rank):
        cdef np.ndarray tmp
        tmp = np.array(self.cInstance.getRecordedValue(post_rank))
        self.cInstance.clearRecordedValue(post_rank)
        return tmp

    # Delay (read-only)
    cpdef np.ndarray _get_delay(self, int post_rank):
        return np.array(self.cInstance.getDelay(post_rank))

    cpdef createFromDict( self, dict dendrites ):
        cdef int rank
        cdef dict data
        for rank, data in dendrites.iteritems():
            # create dendrite instance
            self.cInstance.addDendrite(rank, data['rank'], data['weight'], data['delay'])            
            # initialize variables
            self.cInstance.initValues(rank)

    cpdef createFromCSR( self, dendrites ):
        cdef int rank
        cdef list data
        for rank, data in dendrites.get_data().iteritems():
            # create dendrite instance
            self.cInstance.addDendrite(rank, data[0], data[1], data[2])            
            # initialize variables
            self.cInstance.initValues(rank)

    cpdef int _nb_dendrites(self):
        return self.cInstance.nbDendrites()

    cpdef int _nb_synapses(self, int post_rank):
        return self.cInstance.nbSynapses(post_rank)

    cpdef bool _get_learning(self):
        return self.cInstance.isLearning()
        
    cpdef _set_learning(self, bool learning):
        self.cInstance.setLearning(learning)        

    cpdef int _get_learn_frequency(self):
        return self.cInstance.getLearnFrequency()
        
    cpdef _set_learn_frequency(self, int frequency):
        self.cInstance.setLearnFrequency(frequency)        

    cpdef int _get_learn_offset(self):
        return self.cInstance.getLearnOffset()
        
    cpdef _set_learn_offset(self, int offset):
        self.cInstance.setLearnOffset(offset)        
    
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
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector import CSR as CSR

cdef extern from "../build/%(name)s.h":
    cdef cppclass %(name)s:
        %(name)s(int preLayer, int postLayer, int target)

        void addDendrite(int, vector[int], vector[float], vector[int])

        void addSynapse(int, int, float, int)

        void removeSynapse(int, int)

        void initValues(int post_rank)
         
        vector[int] getRank(int post_rank)

        # local variable value
        vector[float] getValue(int post_rank)
        void setValue(int post_rank, vector[float] values)
        void startRecordValue(int post_rank)
        void stopRecordValue(int post_rank)
        void clearRecordedValue(int post_rank)
        vector[vector[float]] getRecordedValue(int post_rank)
        
        vector[int] getDelay(int post_rank)
        
        int nbDendrites()

        int nbSynapses(int post_rank)

        void setLearning(bool)
    
        void setLearnFrequency(int)
    
        int getLearnFrequency()
    
        void setLearnOffset( int )
    
        int getLearnOffset( )
    
        bool isLearning()
                
%(cFunction)s

cdef class py%(name)s:

    cdef %(name)s* cInstance

    def __cinit__(self, preID, postID, target):
        self.cInstance = new %(name)s(preID, postID, target)

    cpdef add_synapse(self, int post_rank, int pre_rank, float value, int delay):
        self.cInstance.addSynapse(post_rank, pre_rank, value, delay)

    cpdef remove_synapse(self, int post_rank, int pre_rank):
        self.cInstance.removeSynapse(post_rank, pre_rank)

    # Rank (read only)
    cpdef np.ndarray _get_rank(self, int post_rank):
        return np.array(self.cInstance.getRank(post_rank))

    # Value
    cpdef np.ndarray _get_value(self, int post_rank):
        return np.array(self.cInstance.getValue(post_rank))

    cpdef void _set_value(self, int post_rank, np.ndarray value ):
        self.cInstance.setValue(post_rank, value)

    def _start_record_value(self, int post_rank):
        self.cInstance.startRecordValue(post_rank)

    def _stop_record_value(self, int post_rank):
        self.cInstance.stopRecordValue(post_rank)

    cpdef np.ndarray _get_recorded_value(self, int post_rank):
        cdef np.ndarray tmp
        tmp = np.array(self.cInstance.getRecordedValue(post_rank))
        self.cInstance.clearRecordedValue(post_rank)
        return tmp

    # Delay (read-only)
    cpdef np.ndarray _get_delay(self, int post_rank):
        return np.array(self.cInstance.getDelay(post_rank))

    cpdef createFromDict( self, dict dendrites ):
        cdef int rank
        cdef dict data        
        for rank, data in dendrites.iteritems():
            # create dendrite instance
            self.cInstance.addDendrite(rank, data['rank'], data['weight'], data['delay'])            
            # initialize variables
            self.cInstance.initValues(rank)

    cpdef createFromCSR( self, dendrites ):
        cdef int rank
        cdef list data
        for rank, data in dendrites.get_data().iteritems():
            # create dendrite instance
            self.cInstance.addDendrite(rank, data[0], data[1], data[2])            
            # initialize variables
            self.cInstance.initValues(rank)

    cpdef int _nb_dendrites(self):
        return self.cInstance.nbDendrites()

    cpdef int _nb_synapses(self, int post_rank):
        return self.cInstance.nbSynapses(post_rank)
        
    cpdef bool _get_learning(self):
        return self.cInstance.isLearning()
        
    cpdef _set_learning(self, bool learning):
        self.cInstance.setLearning(learning)        

    cpdef int _get_learn_frequency(self):
        return self.cInstance.getLearnFrequency()
        
    cpdef _set_learn_frequency(self, int frequency):
        self.cInstance.setLearnFrequency(frequency)        

    cpdef int _get_learn_offset(self):
        return self.cInstance.getLearnOffset()
        
    cpdef _set_learn_offset(self, int offset):
        self.cInstance.setLearnOffset(offset)        
        
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

    # local: %(name)s
    cpdef np.ndarray _get_%(name)s(self, int post_rank):
        return np.array(self.cInstance.get%(Name)s(post_rank))
        
    cpdef _set_%(name)s(self, int post_rank, np.ndarray value):
        self.cInstance.set%(Name)s(post_rank, value)

    cpdef %(type)s _get_single_%(name)s(self, int post_rank, int rank):
        return self.cInstance.getSingle%(Name)s(post_rank, rank)

    cpdef _set_single_%(name)s(self, int post_rank, int rank, %(type)s value):
        self.cInstance.setSingle%(Name)s(post_rank, rank, value)

    def _start_record_%(name)s(self, int post_rank):
        self.cInstance.startRecord%(Name)s(post_rank)

    def _stop_record_%(name)s(self, int post_rank):
        self.cInstance.stopRecord%(Name)s(post_rank)

    cpdef np.ndarray _get_recorded_%(name)s(self, int post_rank):
        tmp = np.array(self.cInstance.getRecorded%(Name)s(post_rank))
        self.cInstance.clearRecorded%(Name)s(post_rank)
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
    cpdef %(type)s _get_%(name)s(self, int post_rank):
        return self.cInstance.get%(Name)s(post_rank)
        
    cpdef _set_%(name)s(self, int post_rank, %(type)s value):
        self.cInstance.set%(Name)s(post_rank, value)
        
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
        vector[%(type)s] get%(Name)s(int post_rank)
        void set%(Name)s(int post_rank, vector[%(type)s] values)
        %(type)s getSingle%(Name)s(int post_rank, int rank)
        void setSingle%(Name)s(int post_rank, int rank, %(type)s values)
        void startRecord%(Name)s(int post_rank)
        void stopRecord%(Name)s(int post_rank)
        void clearRecorded%(Name)s(int post_rank)
        vector[vector[%(type)s]] getRecorded%(Name)s(int post_rank)
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
        %(type)s get%(Name)s(int post_rank)
        void set%(Name)s(int post_rank, %(type)s value)                
"""

# an implementation code of adding dynamically synapses.
# 
# Depends on:
# 
#     * add_synapse : push_back of local variables, except value, delay, rank
add_synapse_body = """
    for(unsigned int i=0; i < rank_.size(); i++) 
    {
        if(rank_[i] == rank ) 
        {
        #ifdef _DEBUG
            std::cout << "synapse already exists ... " << std::endl;
        #endif
            return -1;
        }
    }

    rank_.push_back(rank);
    value_.push_back(value);
    
    if( delay > 0 )
    {
        delay_.push_back(delay);
        if(delay > maxDelay_)
        {
            maxDelay_ = delay;
            pre_population_->setMaxDelay(maxDelay_);
        }
    }

    if ( !isRateCoded_ )
    {
        auto tmp = std::pair<int,int>(rank, rank_.size()-1);
        inv_rank_.insert( tmp );
    }
    
    %(add_synapse)s

    nbSynapses_++;
    return 0;
"""

# an implementation code of removing dynamically synapses.
# 
# Depends on:
# 
#     * rem_synapse : erase of local variables, except value, delay, rank
rem_synapse_body = """
#ifdef _DEBUG
    std::cout << "suppress synapse - pre = " << rank << std::endl;
    std::cout << "check "<< rank_.size() <<" synapses."<< std::endl;
#endif
    for(unsigned int i=0; i < rank_.size(); i++) 
    {
        if(rank_[i] == rank ) 
        {
        #ifdef _DEBUG
           std::cout << "found the synapse at: "<< i <<std::endl;
        #endif
        
           rank_.erase(rank_.begin()+i);
           value_.erase(value_.begin()+i);

           if (delay_.size() > 1)
               delay_.erase(delay_.begin()+i);

            %(rem_synapse)s

           nbSynapses_--;
           return 0;
        }
    }

    return -1;
"""

rem_all_synapse_body = """
    rank_.clear();
    value_.clear();
    delay_.clear();
    
    nbSynapses_ = 0;
    
    %(rem_all_synapse)s
    
    if( !isRateCoded_ )
        inv_rank_.clear();
"""
