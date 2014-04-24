
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
#include "RateDendrite.h"

class %(class)s : public RateDendrite 
{
public:
    %(class)s(Population* pre, Population* post, int postRank, int target);
    
    %(class)s(int preID, int postID, int postRank, int target);
    
    ~%(class)s();
    
    class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    int addSynapse(int rank, DATA_TYPE value, int delay);

    int removeSynapse(int rank);
    
    int removeAllSynapses();
    
    void initValues();
    
    void computeSum();
    
    void globalLearn();
    
    void localLearn();

    void record();

%(access)s

%(functions)s

private:
%(member)s

    %(pre_name)s* pre_population_;
    %(post_name)s* post_population_;
};
#endif
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
#include "SpikeDendrite.h"

class %(class)s : public SpikeDendrite 
{
public:
    %(class)s(Population* pre, Population* post, int postRank, int target);
    
    %(class)s(int preID, int postID, int postRank, int target);
    
    ~%(class)s();
    
    class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }

    int addSynapse(int rank, DATA_TYPE value, int delay);

    int removeSynapse(int rank);
    
    int removeAllSynapses();
    
    void initValues();
    
    void computeSum();
    
    void globalLearn();
    
    void localLearn();

    void preEvent(int rank);
    
    void postEvent();
    
    bool isPreSynaptic(Population* pop) { return pop == static_cast<Population*>(pre_population_); }
        
    void record();
    
%(access)s

%(functions)s

private:
%(member)s

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

using namespace ANNarchy_Global;

%(class)s::%(class)s(Population* pre, Population* post, int postRank, int target) : RateDendrite() 
{
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);
    
    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addDendrite(postRank, this);
}

%(class)s::%(class)s(int preID, int postID, int postRank, int target) : RateDendrite() 
{
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    pre_rates_ = pre_population_->getRates();
    post_rates_ = post_population_->getRates();

    target_ = target;
    post_neuron_rank_ = postRank;

    post_population_->addDendrite(postRank, this);
}

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout <<"%(class)s::Destructor"<< std::endl;
#endif

%(destructor)s
}

int %(class)s::addSynapse(int rank, DATA_TYPE value, int delay)
{
%(add_synapse_body)s
}

int %(class)s::removeSynapse(int rank)
{
%(rem_synapse_body)s
}

int %(class)s::removeAllSynapses()
{
%(rem_all_synapse_body)s
}

void %(class)s::initValues() 
{
%(init)s
}

void %(class)s::computeSum() 
{   
%(sum)s
}

void %(class)s::localLearn() 
{
%(local)s
}

void %(class)s::globalLearn() 
{
%(global)s
}

void %(class)s::record() 
{
%(record)s
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
#    * local : code for local_learn 
#
#    * global : code for global_learn
#
spike_projection_body = \
"""#include "%(class)s.h"        
#include "Global.h"

using namespace ANNarchy_Global;
        
%(class)s::%(class)s(Population* pre, Population* post, int postRank, int target) : SpikeDendrite() 
{
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);

    target_ = target;
    post_neuron_rank_ = postRank;
    
    post_population_->addDendrite(postRank, this);
    pre_population_->addSpikeTarget(this);
}

%(class)s::%(class)s(int preID, int postID, int postRank, int target) : SpikeDendrite() 
{
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    target_ = target;
    post_neuron_rank_ = postRank;
    
    post_population_->addDendrite(postRank, this);
    pre_population_->addSpikeTarget(this);
}

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout<<"%(class)s::Destructor"<<std::endl;
#endif

%(destructor)s
}

int %(class)s::addSynapse(int rank, DATA_TYPE value, int delay)
{
%(add_synapse_body)s
}

int %(class)s::removeSynapse(int rank)
{
%(rem_synapse_body)s
}

int %(class)s::removeAllSynapses()
{
%(rem_all_synapse_body)s
}

void %(class)s::initValues() 
{
%(init)s
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

void %(class)s::record() 
{
%(record)s
}

void %(class)s::preEvent(int rank) 
{
%(pre_event)s
}

void %(class)s::postEvent() 
{
%(post_event)s
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
    #ifdef _DEBUG
        std::cout << "sum over " << nbSynapses_ << " elements." << std::endl;
    #endif
        for(int i=0; i < nbSynapses_; i++) 
        {
            sum_ += %(psp)s
        }        
    }
    else    // delayed connections
    {
        if(constDelay_) // one delay for all connections
        {
            pre_rates_ = static_cast<RatePopulation*>(pre_population_)->getRates(delay_[0]);
            
        #ifdef _DEBUG
            std::cout << "pre_rates_: " << (*pre_rates_).size() << "("<< pre_rates_ << "), for delay " << delay_[0] << std::endl;
            for(int i=0; i<(int)(*pre_rates_).size(); i++) 
            {
                std::cout << (*pre_rates_)[i] << " ";
            }
            std::cout << std::endl;
        #endif
            
            for(int i=0; i < nbSynapses_; i++) 
            {
                sum_ += %(psp_const_delay)s
            }
        }
        else    // different delays [0..maxDelay]
        {
            std::vector<DATA_TYPE> delayedRates = static_cast<RatePopulation*>(pre_population_)->getRates(delay_, rank_);

            for(int i=0; i < nbSynapses_; i++) 
            {
                sum_ += %(psp_dyn_delay)s
            }
        }
    }
    
    //std::cout << "sum(CPU): " << sum_ << std::endl;
""" 

# Template for the preEvent() method of a projection
#
# * eq: equations for the update 
#
pre_event_body="""
#ifdef _DEBUG
    std::cout << "Emitted a pre-synaptic event: "<< rank << " to " << post_neuron_rank_  << std::endl;
    std::cout << "Pre: " << pre_population_->getName() << ", neuron = "<< rank << std::endl;
    std::cout << "Post: " << post_population_->getName() << ", neuron = " << post_neuron_rank_ << std::endl;
#endif

%(eq)s
"""

# Template for the postEvent() method of a projection
#
# * eq: equations for the update
#
post_event_body="""
#ifdef _DEBUG
    std::cout << "Emitted a post-synaptic event" << std::endl;
#endif
    for(int i = 0; i < rank_.size(); i++)
    {
%(eq)s
    }
    
"""
