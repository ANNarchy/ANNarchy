/*
 *    Network.cpp
 *
 *    This file is part of ANNarchy.
 *   
 *   Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
 *   Helge Ülo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   ANNarchy is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "Network.h"

Network::Network() {
	populations_.clear();

	time_ = 0;
}

Network::~Network() {

    std::cout << "Network destructor." << std::endl;

	while(!populations_.empty()) {
		delete populations_.back();
		populations_.pop_back();
	}
}

class Population* Network::getPopulation(std::string name) {
	for(auto it = populations_.cbegin(); it != populations_.end(); it++) {
		if((*it)->getName().compare(name)==0)
			return (*it);
	}

	return NULL;
}

std::vector<DATA_TYPE> Network::getRates(int populationID) {
	return *(populations_[populationID]->getRates());
}

std::vector<DATA_TYPE> Network::getRates(int populationID,int delay) {
	return *(populations_[populationID]->getRates(delay));
}

std::vector<DATA_TYPE> Network::getRates(int populationID, std::vector<int> delays, std::vector<int> ranks) {
	return populations_[populationID]->getRates(delays, ranks);
}

void Network::addPopulation(class Population* population) {
#ifdef _DEBUG
    std::cout << "Added population '"<< population->getName()<<"' on place " << populations_.size()<<std::endl;
#endif
    populations_.push_back(population);
}

void Network::connect(int prePopulationID, int postPopulationID, Connector *connector, int projectionID, int targetID) {
	connector->connect(populations_[prePopulationID], populations_[postPopulationID], projectionID, targetID);
}

void Network::disconnect(int prePopulationID, int postPopulationID) {
	populations_[postPopulationID]->removeProjection(populations_[prePopulationID]);
}

void Network::run(int steps) {

	for(int i =0; i<steps; i++) {
	    // update time in all populations
	    for(int p=0; p<(int)populations_.size(); p++) {
	        populations_[p]->time_ = time_;
	    }

		for(int p=0; p<(int)populations_.size(); p++) {
			populations_[p]->metaSum();
		}

		for(int p=0; p<(int)populations_.size(); p++)
			populations_[p]->metaStep();

		for(int p=0; p<(int)populations_.size(); p++)
			populations_[p]->globalOperations();

		for(int p=0; p<(int)populations_.size(); p++) {
			populations_[p]->metaLearn();
		}

		for(int p=0; p<(int)populations_.size(); p++) {
			populations_[p]->record();
		}

		time_++;
	}

	/*
	double start, stop;
	
	char buffer1[40];
	char buffer2[80];
	FILE *f =NULL;
	FILE *f2=NULL;
#ifdef _WIN32
	sprintf_s(buffer1, 40*sizeof(char), "profile n=%05i c=%04i cs=%04i/", populations_[0]->getNeuronCount(), (populations_[0]->getProjection(0,0))->getWeightCount(), chunkSize);
	CreateDirectory(buffer1, NULL);
	sprintf_s(buffer2, 80*sizeof(char), "%s/sum(%02i).txt", buffer1, omp_get_max_threads());
	fopen_s(&f, buffer2, "a+");
	sprintf_s(buffer2, 80*sizeof(char), "%s/learn(%02i).txt", buffer1, omp_get_max_threads());
	fopen_s(&f2, buffer2, "a+");
#else
	sprintf(buffer1, "profile n=%05i c=%04i cs=%04i\0", populations_[0]->getNeuronCount(), (populations_[0]->getProjection(0,0))->getWeightCount(), chunkSize);
	mkdir(buffer1, 0777);
	sprintf(buffer2, "%s/sum(%02i).txt", buffer1, omp_get_max_threads());
	f = fopen(buffer2, "a+");
	sprintf(buffer2, "%s/learn(%02i).txt", buffer1, omp_get_max_threads());
	f2 = fopen(buffer2, "a+");
#endif
	if(!f || !f2) {
		printf("Couldn't create file => abort.\n");
		return;
	}

	for(int i =0; i<steps; i++) {
		
		for(int p=0; p<(int)populations_.size(); p++) {
			start = omp_get_wtime();		
			populations_[p]->metaSum();
			stop = omp_get_wtime();
			fprintf(f, "%f ",(stop-start)*1000);
		}
		fprintf(f, "\n");
		
		for(int p=0; p<(int)populations_.size(); p++)
			populations_[p]->metaStep();

		for(int p=0; p<(int)populations_.size(); p++) {
			start = omp_get_wtime();		
			populations_[p]->metaLearn();
			stop = omp_get_wtime();
			fprintf(f2, "%f ",(stop-start)*1000);
		}
		fprintf(f2, "\n");
	}
	fprintf(f, "\n");
	fprintf(f2, "\n");

	fclose(f);
	fclose(f2);
	*/
}
