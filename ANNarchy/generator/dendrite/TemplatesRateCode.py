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
rate_dendrite_header = \
"""#ifndef __%(class)s_H__
#define __%(class)s_H__

#include "Global.h"
#include "Includes.h"
#include "RateDendrite.h"

class %(class)s : public RateDendrite 
{
public:
    %(class)s(Population* pre, Population* post, int postRank, int target, class RateProjection* proj);
    
    %(class)s(int preID, int postID, int postRank, int target, class RateProjection* proj);
    
    ~%(class)s();
    
    class Population* getPrePopulation() { return static_cast<Population*>(pre_population_); }
    
    int getSynapseCount() { return nbSynapses_; }
    
    int addSynapse(int rank, DATA_TYPE w, int delay);

    int removeSynapse(int rank);
    
    int removeAllSynapses();
    
    void computeSum();
    
    void globalLearn();
    
    void localLearn();

    void record();

%(access)s

%(functions)s

private:
%(member)s

%(random)s

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
rate_dendrite_body = \
"""#include "%(class)s.h"        
#include "Global.h"

%(add_include)s

using namespace ANNarchy_Global;

%(class)s::%(class)s(Population* pre, Population* post, int postRank, int target, class RateProjection* proj) : RateDendrite(proj) 
{
#ifdef _DEBUG
    std::cout << "Create %(class)s (ptr = " << this << ") for neuron " << postRank << " between " << pre->getName() << " and " << post->getName() << std::endl;
#endif
    pre_population_ = static_cast<%(pre_type)s*>(pre);
    post_population_ = static_cast<%(post_type)s*>(post);
    
    pre_rates_ = pre_population_->getRs();
    post_rates_ = post_population_->getRs();

#ifdef _DEBUG
    std::cout << "pre_rates_ = " << pre_rates_ << ", post_rates_ = " << post_rates_ << std::endl;  
#endif    

    target_ = target;
    post_neuron_rank_ = postRank;
    
%(constructor)s
}

%(class)s::%(class)s(int preID, int postID, int postRank, int target, class RateProjection* proj) : RateDendrite(proj) 
{
#ifdef _DEBUG
    std::cout << "Create %(class)s (ptr = " << this << ") for neuron " << postRank << " between id =" << preID << " and id =" << postID << std::endl;
#endif    
    pre_population_ = static_cast<%(pre_type)s*>(Network::instance()->getPopulation(preID));
    post_population_ = static_cast<%(post_type)s*>(Network::instance()->getPopulation(postID));

    pre_rates_ = pre_population_->getRs();
    post_rates_ = post_population_->getRs();

#ifdef _DEBUG
    std::cout << "pre_rates_ = " << pre_rates_ << ", post_rates_ = " << post_rates_ << std::endl;  
#endif    

    target_ = target;
    post_neuron_rank_ = postRank;

    
    // Recording of weights disabled by default
    record_w_ = false;
    
%(constructor)s
}

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout <<"%(class)s::Destructor"<< std::endl;
#endif

%(destructor)s
}

int %(class)s::addSynapse(int rank, DATA_TYPE w, int delay)
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
