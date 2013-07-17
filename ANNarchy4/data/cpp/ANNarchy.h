#ifndef __ANNARCHY_H__
#define __ANNARCHY_H__
    
#include "Includes.h"

class ANNarchy {
public:
    ANNarchy() {
		net_ = Network::instance();

		//AddPopulation

		//AddProjection
	}

    ~ANNarchy() {
        delete net_;
    }

    std::vector<DATA_TYPE> getRates(int populationID) {
	return (*net_->getPopulation(populationID)->getRates());
    }

    void run(int nbSteps) {
        net_->run(nbSteps);
    }
private:
    Network *net_;
};

//createProjInstance

#endif
