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

    std::vector<int> get_rank(int post_rank);
    
    std::vector<DATA_TYPE> get_w(int post_rank);    
    void set_w(int post_rank, std::vector<DATA_TYPE> values);
    
    std::vector<int> get_delay(int post_rank);
    
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
#include "ParallelLogger.h"
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
#ifdef _DEBUG_PARALLELISM
    log_->resize(nbDendrites_);
#endif    
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

std::vector<int> %(class)s::get_rank(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_rank();
}

std::vector<DATA_TYPE> %(class)s::get_w(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_w();
}

void %(class)s::set_w(int post_rank, std::vector<DATA_TYPE> values)
{
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->set_w(values);
}

std::vector<int> %(class)s::get_delay(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_delay();
}

%(access)s

void %(class)s::addDendrite(int post_rank, std::vector<int> rank, std::vector<DATA_TYPE> w, std::vector<int> delay)
{
    dendrites_[post_rank] = static_cast<RateDendrite*>(new %(dend_class)s(pre_population_, post_population_, post_rank, target_));
    
    dendrites_[post_rank]->set_rank(rank);
    dendrites_[post_rank]->set_w(w);
    dendrites_[post_rank]->set_delay(delay);    
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

    std::vector<int> get_rank(int post_rank);
    
    std::vector<DATA_TYPE> get_w(int post_rank);
    void set_w(int post_rank, std::vector<DATA_TYPE> values);
    
    std::vector<int> get_delay(int post_rank);
    
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

std::vector<int> %(class)s::get_rank(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_rank();
}

std::vector<DATA_TYPE> %(class)s::get_w(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_w();
}

void %(class)s::set_w(int post_rank, std::vector<DATA_TYPE> values)
{
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->set_w(values);
}

std::vector<int> %(class)s::get_delay(int post_rank)
{
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_delay();
}

%(access)s

void %(class)s::addDendrite(int post_rank, std::vector<int> rank, std::vector<DATA_TYPE> w, std::vector<int> delay)
{
    dendrites_[post_rank] = static_cast<SpikeDendrite*>(new %(dend_class)s(pre_population_, post_population_, post_rank, target_));
    
    dendrites_[post_rank]->set_rank(rank);
    dendrites_[post_rank]->set_w(w);
    dendrites_[post_rank]->set_delay(delay);    
}
"""

# Template for a local variable
# 
# Depends on:
# 
#     * name : name of the variable
#
#     * type : type of the variable
#
local_idx_variable_access = \
"""
    // Access methods for the local variable %(name)s
    std::vector<%(type)s> get_%(name)s(int post_rank);
    void set_%(name)s(int post_rank, std::vector<%(type)s> %(name)s);
    
    %(type)s get_single_%(name)s(int post_rank, int rank);
    void set_single_%(name)s(int post_rank, int rank, %(type)s %(name)s);

    std::vector< std::vector< %(type)s > >get_recorded_%(name)s(int post_rank);                    
    void start_record_%(name)s(int post_rank);
    void stop_record_%(name)s(int post_rank);
    void clear_recorded_%(name)s(int post_rank);
"""

local_idx_variable_access_body = \
"""
// Access methods for the local variable %(name)s
std::vector<%(type)s> %(class)s::get_%(name)s(int post_rank) 
{     
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_%(name)s(); 
}

void %(class)s::set_%(name)s(int post_rank, std::vector<%(type)s> %(name)s) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->set_%(name)s(%(name)s); 
}

%(type)s %(class)s::get_single_%(name)s(int post_rank, int rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_single_%(name)s(rank); 
}
void %(class)s::set_single_%(name)s(int post_rank, int rank, %(type)s %(name)s) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->set_single_%(name)s(rank, %(name)s); 
}

std::vector< std::vector< %(type)s > > %(class)s::get_recorded_%(name)s(int post_rank) 
{ 
    return (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->get_recorded_%(name)s(); 
}                    
void %(class)s::start_record_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->start_record_%(name)s(); 
}
void %(class)s::stop_record_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->stop_record_%(name)s(); 
}
void %(class)s::clear_recorded_%(name)s(int post_rank) 
{ 
    (static_cast<%(dend_class)s*>(dendrites_[post_rank]))->clear_recorded_%(name)s(); 
}
"""

# Template for a global variable
# 
# Depends on:
# 
#     * name : name of the variable
#
#     * type : type of the variable
#
global_idx_variable_access = \
"""
    // Access methods for the global variable %(name)s
    %(type)s get_%(name)s(int post_rank);
    void set_%(name)s(int post_rank, %(type)s %(name)s);
"""

global_idx_variable_access_body = \
"""
    // Access methods for the global variable %(name)s
    %(type)s %(class)s::get_%(name)s(int post_rank) 
    {  
        return (static_cast<class %(dend_class)s*>(dendrites_[post_rank]))->get_%(name)s(); 
    }
    
    void %(class)s::set_%(name)s(int post_rank, %(type)s %(name)s) 
    { 
        (static_cast<class %(dend_class)s*>(dendrites_[post_rank]))->set_%(name)s( %(name)s ); 
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
         
        vector[int] get_rank(int post_rank)

        # local variable w
        vector[float] get_w(int post_rank)
        void set_w(int post_rank, vector[float] values)
        void start_record_w(int post_rank)
        void stop_record_w(int post_rank)
        void clear_recorded_w(int post_rank)
        vector[vector[float]] get_recorded_w(int post_rank)
        
        vector[int] get_delay(int post_rank)
        
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

    cpdef add_synapse(self, int post_rank, int pre_rank, float w, int delay):
        self.cInstance.addSynapse(post_rank, pre_rank, w, delay)

    cpdef remove_synapse(self, int post_rank, int pre_rank):
        self.cInstance.removeSynapse(post_rank, pre_rank)

    # Rank (read only)
    cpdef np.ndarray _get_rank(self, int post_rank):
        return np.array(self.cInstance.get_rank(post_rank))

    # w
    cpdef np.ndarray _get_w(self, int post_rank):
        return np.array(self.cInstance.get_w(post_rank))

    cpdef _set_w(self, int post_rank, np.ndarray value ):
        self.cInstance.set_w(post_rank, value)

    def _start_record_w(self, int post_rank):
        self.cInstance.start_record_w(post_rank)

    def _stop_record_w(self, int post_rank):
        self.cInstance.stop_record_w(post_rank)

    cpdef np.ndarray _get_recorded_w(self, int post_rank):
        cdef np.ndarray tmp
        tmp = np.array(self.cInstance.get_recorded_w(post_rank))
        self.cInstance.clear_recorded_w(post_rank)
        return tmp

    # Delay (read-only)
    cpdef np.ndarray _get_delay(self, int post_rank):
        return np.array(self.cInstance.get_delay(post_rank))

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
        cdef dict delays 

        delays = dendrites.get_delay()
        for rank, data in dendrites.get_data().iteritems():

            # create dendrite instance
            self.cInstance.addDendrite(rank, data[0], data[1], delays[rank])            
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
         
        vector[int] get_rank(int post_rank)

        # local variable w
        vector[float] get_w(int post_rank)
        void set_w(int post_rank, vector[float] values)
        void start_record_w(int post_rank)
        void stop_record_w(int post_rank)
        void clear_recorded_w(int post_rank)
        vector[vector[float]] get_recorded_w(int post_rank)
        
        vector[int] get_delay(int post_rank)
        
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

    cpdef add_synapse(self, int post_rank, int pre_rank, float w, int delay):
        self.cInstance.addSynapse(post_rank, pre_rank, w, delay)

    cpdef remove_synapse(self, int post_rank, int pre_rank):
        self.cInstance.removeSynapse(post_rank, pre_rank)

    # Rank (read only)
    cpdef np.ndarray _get_rank(self, int post_rank):
        return np.array(self.cInstance.get_rank(post_rank))

    # w
    cpdef np.ndarray _get_w(self, int post_rank):
        return np.array(self.cInstance.get_w(post_rank))

    cpdef _set_w(self, int post_rank, np.ndarray value ):
        self.cInstance.set_w(post_rank, value)

    def _start_record_w(self, int post_rank):
        self.cInstance.start_record_w(post_rank)

    def _stop_record_w(self, int post_rank):
        self.cInstance.stop_record_w(post_rank)

    cpdef np.ndarray _get_recorded_w(self, int post_rank):
        cdef np.ndarray tmp
        tmp = np.array(self.cInstance.get_recorded_w(post_rank))
        self.cInstance.clear_recorded_w(post_rank)
        return tmp

    # Delay (read-only)
    cpdef np.ndarray _get_delay(self, int post_rank):
        return np.array(self.cInstance.get_delay(post_rank))

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
        cdef dict delays 

        delays = dendrites.get_delay()
        for rank, data in dendrites.get_data().iteritems():

            # create dendrite instance
            self.cInstance.addDendrite(rank, data[0], data[1], delays[rank])            
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
local_property_pyx = """

    # local: %(name)s
    cpdef np.ndarray _get_%(name)s(self, int post_rank):
        return np.array(self.cInstance.get_%(name)s(post_rank))
        
    cpdef _set_%(name)s(self, int post_rank, np.ndarray value):
        self.cInstance.set_%(name)s(post_rank, value)

    cpdef %(type)s _get_single_%(name)s(self, int post_rank, int rank):
        return self.cInstance.get_single_%(name)s(post_rank, rank)

    cpdef _set_single_%(name)s(self, int post_rank, int rank, %(type)s value):
        self.cInstance.set_single_%(name)s(post_rank, rank, value)

    def _start_record_%(name)s(self, int post_rank):
        self.cInstance.start_record_%(name)s(post_rank)

    def _stop_record_%(name)s(self, int post_rank):
        self.cInstance.stop_record_%(name)s(post_rank)

    cpdef np.ndarray _get_recorded_%(name)s(self, int post_rank):
        tmp = np.array(self.cInstance.get_recorded_%(name)s(post_rank))
        self.cInstance.clear_recorded_%(name)s(post_rank)
        return tmp
        
"""

# Global Cython property
# 
# Depends on:
# 
#     * name : name of the variable
global_property_pyx = """
    
    # global: %(name)s
    cpdef %(type)s _get_%(name)s(self, int post_rank):
        return self.cInstance.get_%(name)s(post_rank)
        
    cpdef _set_%(name)s(self, int post_rank, %(type)s value):
        self.cInstance.set_%(name)s(post_rank, value)
        
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
        vector[%(type)s] get_%(name)s(int post_rank)
        void set_%(name)s(int post_rank, vector[%(type)s] values)
        %(type)s get_single_%(name)s(int post_rank, int rank)
        void set_single_%(name)s(int post_rank, int rank, %(type)s values)
        void start_record_%(name)s(int post_rank)
        void stop_record_%(name)s(int post_rank)
        void clear_recorded_%(name)s(int post_rank)
        vector[vector[%(type)s]] get_recorded_%(name)s(int post_rank)
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
        %(type)s get_%(name)s(int post_rank)
        void set_%(name)s(int post_rank, %(type)s value)                
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
    w_.push_back(w);
    
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
#     * rem_synapse : erase of local variables, except w, delay, rank
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
           w_.erase(w_.begin()+i);

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
    w_.clear();
    delay_.clear();
    
    nbSynapses_ = 0;
    
    %(rem_all_synapse)s
    
    if( !isRateCoded_ )
        inv_rank_.clear();
"""
