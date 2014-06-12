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
    
    if( delay > dt_ )
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