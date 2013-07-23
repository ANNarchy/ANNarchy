#include "Connector.h"
#include "Includes.h"

#include <iostream>

Projection* Connector::instantiateProj(int projectionID, Population *prePopulation, Population *postPopulation, int postID, int target ) {
	return createProjInstance().getInstanceOf(projectionID, prePopulation, postPopulation, postID, target);
}

All2AllConnector::All2AllConnector(bool allowSelfConnections, Distribution<DATA_TYPE> *weight, Distribution<int> *delay) {
	allowSelfConnections_ = allowSelfConnections;
	weight_ = weight;
	delay_ = delay;

#ifdef _DEBUG
	std::cout << "Instantiate All2AllConnector" << std::endl;
#endif
}

All2AllConnector::~All2AllConnector() {
#ifdef _DEBUG
	std::cout << "Remove All2AllConnector" << std::endl;
#endif
	delete weight_;
	delete delay_;
}

void All2AllConnector::connect(Population *prePopulation, Population *postPopulation, int projectionID, int target) {
#ifdef _DEBUG
	std::cout << "All2AllConnector" << std::endl;
	std::cout << "	PreLayer -" << prePopulation->getName() << prePopulation->getNeuronCount() << std::endl;
	std::cout << "	PostLayer -" << prePopulation->getName() << postPopulation->getNeuronCount() << std::endl;
#endif

	std::vector<int> ranks;

	if((prePopulation == postPopulation) && (!allowSelfConnections_)) {
		for(int postID=0; postID<postPopulation->getNeuronCount(); postID++) {
			ranks.clear();

			// generate pre synaptic ranks
			for(int preID=0; preID<prePopulation->getNeuronCount(); preID++) {
				if(preID != postID) {
					ranks.push_back(preID);
				}
			}

			// add projection
			Projection *proj = instantiateProj(projectionID, prePopulation, postPopulation, postID, target);
			if(delay_)
				proj->initValues(ranks, weight_->getValues(prePopulation->getNeuronCount()-1), delay_->getValues(prePopulation->getNeuronCount()-1));
			else
				proj->initValues(ranks, weight_->getValues(prePopulation->getNeuronCount()-1));
			postPopulation->addProjection(postID, proj);
		}
	}else{
		//
		//	all rank matrices are completely the same
		for(int preID=0; preID<prePopulation->getNeuronCount(); preID++) {
			ranks.push_back(preID);
		}

		// add projection
		for(int postID=0; postID<postPopulation->getNeuronCount(); postID++) {
			Projection *proj = instantiateProj(projectionID, prePopulation, postPopulation, postID, target);
			if(delay_)
				proj->initValues(ranks, weight_->getValues(prePopulation->getNeuronCount()), delay_->getValues(prePopulation->getNeuronCount()));
			else
				proj->initValues(ranks, weight_->getValues(prePopulation->getNeuronCount()));
			postPopulation->addProjection(postID, proj);
		}

	}

}


One2OneConnector::One2OneConnector(Distribution<DATA_TYPE> *weight, Distribution<int> *delay) {
	weight_ = weight;
	delay_ = delay;
}

One2OneConnector::~One2OneConnector() {
	delete weight_;
	delete delay_;
}

void One2OneConnector::connect(Population *prePopulation, Population *postPopulation, int projectionID, int target) {
#ifdef _DEBUG
	std::cout << "One2OneConnector" << std::endl;
	std::cout << "	PreLayer -" << prePopulation->getName() << prePopulation->getNeuronCount() << std::endl;
	std::cout << "	PostLayer -" << prePopulation->getName() << postPopulation->getNeuronCount() << std::endl;
#endif

	std::vector<int> ranks;

	if(postPopulation->getNeuronCount() != prePopulation->getNeuronCount()) {
		std::cout << "Populations require same size."<< std::endl;
		return;
	}

	for(int postID=0; postID<postPopulation->getNeuronCount(); postID++) {
		ranks.clear();
		ranks.push_back(postID);

		// add projection
		Projection *proj = instantiateProj(projectionID, prePopulation, postPopulation, postID, target);
		if(delay_)
			proj->initValues(ranks, weight_->getValues(1), delay_->getValues(1));
		else
			proj->initValues(ranks, weight_->getValues(1));
		postPopulation->addProjection(postID, proj);
	}
}
