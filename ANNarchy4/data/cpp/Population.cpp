/*
 *    Population.cpp
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
#include "Population.h"

Population::Population(std::string name, int nbNeurons) {
	name_ = std::move(name);

	nbNeurons_ = nbNeurons;
	rate_ = std::vector<DATA_TYPE>(nbNeurons_, 0.0);
	projections_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());
	
	maxDelay_ = 0;
	dt_ = 1.0;
	std::vector< std::vector<DATA_TYPE>	> delayedRates_ = std::vector< std::vector<DATA_TYPE> >();

#ifdef ANNAR_PROFILE
    char buffer[200];
    try{
        sprintf(buffer, "%s(%02i)_sum.txt", name_.c_str(), omp_get_max_threads());
        cs = fopen(buffer, "w");
        sprintf(buffer, "%s(%02i)_global.txt", name_.c_str(), omp_get_max_threads());
        gl = fopen(buffer, "w");
        sprintf(buffer, "%s(%02i)_local.txt", name_.c_str(), omp_get_max_threads());
        ll = fopen(buffer, "w");
    }catch(std::exception e){
        std::cout << "Cannnot open file '"<<buffer<<"'" << std::endl;
        std::cout << e.what() << std::endl;
        cs = NULL;
        gl = NULL;
        ll = NULL;
    }
#endif
}

Population::~Population() {
    std::cout << "Population::Destructor" << std::endl;

    rate_.erase(rate_.begin(), rate_.end());
    for(auto it=delayedRates_.begin(); it<delayedRates_.end(); it++)
        (*it).erase((*it).begin(), (*it).end());

    for(int n=0; n<nbNeurons_; n++) {
        while(!projections_[n].empty()){
            delete projections_[n].back();
            projections_[n].pop_back();
        }
        //projections_[n].erase(projections_[n].begin(), projections_[n].end());
    }

#ifdef ANNAR_PROFILE
    if(cs)
        fclose(cs);
    if(gl)
        fclose(gl);
    if(ll)
        fclose(ll);
#endif
}

std::vector<Projection*> Population::getProjections(int neuron, int type) {
    std::vector<Projection*> vec = std::vector<Projection*>();

    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if(projections_[neuron][p]->getTarget() == type)
            {
                vec.push_back(projections_[neuron][p]);
            }
        }
    }

    return vec;
}

Projection* Population::getProjection(int neuron, int type, Population *pre) {
    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if ( (projections_[neuron][p]->getTarget() == type) &&
                 (projections_[neuron][p]->getPrePopulation() == pre) )
            {
                return projections_[neuron][p];
            }
        }
    }

    return NULL;
}

void Population::printRates() {
	for(int n=0; n<nbNeurons_; n++) {
		printf("%.02f ", rate_[n]);
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
#ifdef ANNAR_PROFILE
    double start = omp_get_wtime();
#endif

#ifdef _DEBUG
    std::cout << "###########################"<< std::endl;
    std::cout << "# Meta sum                #"<< std::endl;
    std::cout << "###########################"<< std::endl;
#endif

    #pragma omp parallel for
    for(int n=0; n<nbNeurons_; n++)
    {
    #ifdef _DEBUG
        std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++)
        {
        #ifdef _DEBUG
            std::cout << "reference: " << projections_[n][p] << std::endl;
            std::cout << "\tpost = " << name_ << std::endl;
            std::cout << "\tpre = " << projections_[n][p]->getPrePopulation()->getName() << std::endl;
            std::cout << "\tsynaseCount = " << projections_[n][p]->getSynapseCount() << std::endl;
        #endif
            projections_[n][p]->computeSum();
		}
	}

#ifdef ANNAR_PROFILE
    double stop = omp_get_wtime();

    if(cs)
        fprintf(cs, "%f\n", (stop-start)*1000.0);
#endif
}

void Population::metaStep() {

}

//
// projection update for post neuron based variables
void Population::metaLearn() {
#ifdef ANNAR_PROFILE
    double start = omp_get_wtime();
#endif

#ifdef _DEBUG
    std::cout << "###########################"<< std::endl;
    std::cout << "# Global learning         #"<< std::endl;
    std::cout << "###########################"<< std::endl;
#endif
    #pragma omp parallel for
    for(int n=0; n<nbNeurons_; n++)
    {
    #ifdef _DEBUG
        std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++)
        {
            projections_[n][p]->globalLearn();
        }
    }

#ifdef ANNAR_PROFILE
    double stop = omp_get_wtime();

    if(gl)
        fprintf(gl, "%f\n", (stop-start)*1000.0);
#endif

#ifdef ANNAR_PROFILE
    double start2 = omp_get_wtime();
#endif

#ifdef _DEBUG
    std::cout << "###########################"<< std::endl;
    std::cout << "# Local  learning         #"<< std::endl;
    std::cout << "###########################"<< std::endl;
#endif

    #pragma omp parallel for
    for(int n=0; n<nbNeurons_; n++) {
    #ifdef _DEBUG
        std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->localLearn();
        }
    }

#ifdef ANNAR_PROFILE
    double stop2 = omp_get_wtime();

    if(ll)
        fprintf(ll, "%f\n", (stop2-start2)*1000.0);
#endif
}

void Population::globalOperations() {

}
