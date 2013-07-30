#include "Population.h"

Population::Population(std::string name, int nbNeurons) {
	name_ = std::move(name);

	nbNeurons_ = nbNeurons;
	rate = std::vector<DATA_TYPE>(nbNeurons_, 2.0);
	projections_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());
	
	maxDelay_ = 0;
	dt_ = 1.0;
	std::vector< std::vector<DATA_TYPE>	> delayedRates_ = std::vector< std::vector<DATA_TYPE> >();
}

Population::~Population() {
	rate.clear();
	for(int n=0; n<nbNeurons_; n++) {
		projections_[n].erase(projections_[n].begin(), projections_[n].end());
	}
}

void Population::printRates() {
	for(int n=0; n<nbNeurons_; n++) {
		printf("%.02f ", rate[n]);
		if((n>0)&&(n%10==0))
			printf("\n");
	}
	printf("\n");
}

DATA_TYPE Population::sum(int neur, int typeID) {
	DATA_TYPE sum=0.0;

	for(int i=0; i< projections_[neur].size(); i++)
		if(projections_[neur][i]->getTarget() == typeID)
			sum += projections_[neur][i]->getSum();

	return sum;
}

std::vector<DATA_TYPE> Population::getRates(std::vector<int> delays, std::vector<int> ranks) {
	std::vector<DATA_TYPE> vec = std::vector<DATA_TYPE>(delays.size(), 0.0);

	if(delays.size() != ranks.size()) {
		std::cout << "Invalid vector ranges. " << std::endl;
		return std::vector<DATA_TYPE>();
	}

	for(unsigned int n = 0; n < ranks.size(); n++) {
		vec[n] = delayedRates_[ranks[n]][delays[n]-1];
	}

	return vec;
}

void Population::setMaxDelay(int delay) {
	// TODO:
	// maybe we should take the current fire rate as initial value
	if(delay > maxDelay_) {
		for(int oldSize = delayedRates_.size(); oldSize < delay; oldSize++)
			delayedRates_.push_back(std::vector<DATA_TYPE>(nbNeurons_, (DATA_TYPE)oldSize));
	}
}

void Population::addProjection(int postRankID, Projection* proj) {
#ifdef _DEBUG
	std::cout << name_ << ": added projection to neuron " << postRankID << std::endl;
#endif
	projections_[postRankID].push_back(proj);
}

void Population::removeProjection(Population* pre) {
	for(int n=0; n<nbNeurons_; n++) {
		for(int p=0; p< (int)projections_[n].size();p++) {
			if(projections_[n][p]->getPrePopulation() == pre)
				projections_[n].erase(projections_[n].begin()+p);
		}
	}
}

void Population::metaSum() {

	#pragma omp parallel for schedule(static, 10)
	for(int n=0; n<nbNeurons_; n++) {
		for(int p=0; p< (int)projections_[n].size();p++) {
			projections_[n][p]->computeSum();
		}
	}
}

void Population::metaStep() {

}

void Population::metaLearn() {

    //
    // projection update for post neuron based variables
    #pragma omp parallel for schedule(dynamic, 10)
    for(int n=0; n<nbNeurons_; n++) {
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->globalLearn();
        }
    }

    //
    // projection update for every single synapse
    for(int n=0; n<nbNeurons_; n++) {
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->localLearn();
        }
    }

}
