# Template for the computeSum() method of a projection
#
# * psp: basic code for the psp (default (*pre_rates_)[rank_[i]] * value_[i];) 
#
# * psp_const_delay : code when the delay is constant (normally the same as psp)
#
# * psp_dyn_delay : code when delays are variable (default delayedRates[rank_[i]] * value_[i];) 
#
psp_code_body_omp = \
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
#if defined(_DEBUG) && defined(_DEBUG_PARALLELISM)
    std::cout << "ID of active thread(s) in this block: " << omp_get_thread_num() << std::endl;
#endif
#ifdef _DEBUG
    std::cout << "Evaluate a pre-synaptic event ( time = "<< ANNarchy_Global::time <<"): "<< rank << " to " << post_neuron_rank_  << std::endl;
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
    std::cout << "Evaluate a post-synaptic event" << std::endl;
#endif
    for(int i = 0; i < rank_.size(); i++)
    {
%(eq)s
    }
    
"""
