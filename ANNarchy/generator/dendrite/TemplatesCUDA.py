# Template for the computeSum() method of a projection
#
# * psp: basic code for the psp (default (*pre_rates_)[rank_[i]] * w_[i];) 
#
# * psp_const_delay : code when the delay is constant (normally the same as psp)
#
# * psp_dyn_delay : code when delays are variable (default delayedRates[rank_[i]] * w_[i];) 
#
psp_code_body_cuda = \
"""
    sum_ =0.0;
    
    double start = omp_get_wtime();
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
            pre_rates_ = static_cast<RatePopulation*>(pre_population_)->getR(delay_[0]);
            
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
            std::vector<DATA_TYPE> delayedRates = static_cast<RatePopulation*>(pre_population_)->getR(delay_, rank_);

            for(int i=0; i < nbSynapses_; i++) 
            {
                sum_ += %(psp_dyn_delay)s
            }
        }
    }
    std::cout << "Computation time CPU:"<< (omp_get_wtime() - start)*1000.0 << " ms "<< std::endl;
        
    DATA_TYPE gpu_sum =0.0;
    
    if(delay_.empty() || maxDelay_ == 0)    // no delay
    {
        gpu_sum = weightedSum(rank_, w_, *pre_rates_);
    }
    else    // delayed connections
    {
        if(constDelay_) // one delay for all connections
        {
            pre_rates_ = static_cast<RatePopulation*>(pre_population_)->getR(delay_[0]);
            
            gpu_sum = weightedSum(rank_, w_, *pre_rates_);
        }
        else    // different delays [0..maxDelay]
        {
            std::vector<DATA_TYPE> delayedRates = static_cast<RatePopulation*>(pre_population_)->getR(delay_, rank_);

            gpu_sum = weightedSum(rank_, w_, delayedRates);
        }
    }
    
    std::cout << "sum(CPU): " << sum_ << std::endl;
    std::cout << "sum(GPU): " << gpu_sum << std::endl;
""" 

# Template for the preEvent() method of a projection
#
# * eq: equations for the update 
#
pre_event_body_cuda = """
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
post_event_body_cuda = """
#ifdef _DEBUG
    std::cout << "Emitted a post-synaptic event" << std::endl;
#endif
    for(int i = 0; i < rank_.size(); i++)
    {
%(eq)s
    }
    
"""
