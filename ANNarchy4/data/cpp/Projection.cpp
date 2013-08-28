#include "Projection.h"

Projection::Projection(Population* pre, Population* post, int post_rank, int target) {
#ifdef _DEBUG
    std::cout << "Connect (C1) pre="<<pre->getName()<<" with post=" <<post->getName()<< std::endl;
#endif

    pre_population_ = pre;
	post_population_ = post;

	target_ = target;
	post_neuron_rank_= post_rank;

	sum_ =0.0;
	tau_ =10.0;
	dt_ = 1.0;

	//
	//	TODO:	check the case that the referenced addresses
	//			could get invalid or not.
	pre_rates_ = pre->getRates();
	post_rates_ = post->getRates();
}

//
// called from cython
//
// TODO: 	maybe restructure the call structure, that only the population pointer are taken and then
//			Projection(Population* pre, Population* post, int post_rank, int target)
//			is called.
Projection::Projection(int pre, int post, int post_rank, int target) {
#ifdef _DEBUG
	std::cout << "Connect (C2) pre="<<pre<<" with post=" <<post<< " and: target=" << target << " rank=" << post_rank<< " ptr="<< this << std::endl;
#endif
	pre_population_ = Network::instance()->getPopulation(pre);
    post_population_ = Network::instance()->getPopulation(post);

    target_ = target;
	post_neuron_rank_ = post_rank;

	sum_ = 0.0;
	tau_ = 10.0;
	dt_ = 1.0;

	//
	//	TODO as above^^
	pre_rates_ = pre_population_->getRates();
	post_rates_ = post_population_->getRates();

	//
	// need to register on post population
    post_population_->addProjection(post_rank, this);
}

void Projection::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay){
#ifdef _DEBUG
	std::cout << "update projection: ptr="<< this << " r=" << rank.size() << " v=" << value.size() <<" d=" << delay.size() <<std::endl;
#endif	
    rank_ = rank;
	value_= value;
	delay_= delay;
	constDelay_ = true;
	maxDelay_ = 0;

	if(!delay.empty()) {
		maxDelay_ = delay[0];
		for(auto it=delay.begin(); it != delay.end(); it++) {
			if (*it > maxDelay_ )
				maxDelay_ = *it;

			if (*it != maxDelay_)
				constDelay_ = false;
		}

		pre_population_->setMaxDelay(maxDelay_);
	}
}

int Projection::addSynapse(int rank, DATA_TYPE value, int delay) {
    bool found = false;
    for(unsigned int i=0; i < rank_.size(); i++) {
        if(rank_[i] == rank ) {
            found = true;
            continue;
        }
    }
    
    if(!found){
        rank_.push_back(rank);
        value_.push_back(value);
        delay_.push_back(delay);
        return 0;
    }else{
        return -1;
    }
}
	
int Projection::removeSynapse(int rank) {
#ifdef _DEBUG
    std::cout << "suppress synapse - pre = "<<rank<<std::endl;
    std::cout << "check "<<rank_.size()<<" synapses."<< std::endl;
#endif

    for(unsigned int i=0; i < rank_.size(); i++) {
        if(rank_[i] == rank ) {
           std::cout << "found the synapse at: "<< i <<std::endl;
           rank_.erase(rank_.begin()+i);    
           value_.erase(value_.begin()+i);
           
           if (delay_.size() > 1)
               delay_.erase(delay_.begin()+i);
               
           return 0;
        }
    }
    
    return -1;
}

void Projection::computeSum() {
	sum_ =0.0;

	if(delay_.empty())	// no delay
	{
		for(int w=0; w<(int)rank_.size(); w++) {
			sum_ += (*pre_rates_)[rank_[w]] * value_[w];
		}
	}
	else	// delayed connections
	{
		if(constDelay_) // one delay for all connections
		{
			pre_rates_ = pre_population_->getRates(delay_[0]);

			for(int w=0; w<(int)rank_.size(); w++) {
				sum_ += (*pre_rates_)[rank_[w]] * value_[w];
			}
		}
		else	// different delays [0..maxDelay]
		{
			std::vector<DATA_TYPE> delayedRates = pre_population_->getRates(delay_, rank_);

			for(int w=0; w<(int)rank_.size(); w++) {
				sum_ += (*pre_rates_)[rank_[w]] * value_[w];
			}
		}
	}
}
