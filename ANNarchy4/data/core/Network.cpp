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

#include "MeanPopulation.h"
#include "SpikePopulation.h"

Network::Network()
{
	populations_.clear();

	ANNarchy_Global::time = 0;
}

Network::~Network()
{

    std::cout << "Network destructor." << std::endl;

	while(!populations_.empty())
	{
		delete populations_.back();
		populations_.pop_back();
	}
}

class Population* Network::getPopulation(std::string name)
{
	for(auto it = populations_.cbegin(); it != populations_.cend(); it++)
	{
		if((*it)->getName().compare(name)==0)
			return (*it);
	}

	return NULL;
}

class Population* Network::getPopulation(unsigned int id)
{
    if ( id < populations_.size() )
    {
        return populations_[id];
    }
    else
    {
    #ifdef _DEBUG
        std::cout << "Population id="<<id<<" not exist."<<std::endl;
    #endif
        return NULL;
    }
}

/*
std::vector<DATA_TYPE> Network::getRates(int populationID) {
	return *(populations_[populationID]->getRates());
}

std::vector<DATA_TYPE> Network::getRates(int populationID,int delay) {
	return *(populations_[populationID]->getRates(delay));
}

std::vector<DATA_TYPE> Network::getRates(int populationID, std::vector<int> delays, std::vector<int> ranks) {
	return populations_[populationID]->getRates(delays, ranks);
}
*/

void Network::addPopulation(class Population* population) {
    populations_.push_back(population);

    if(population->isMeanRateCoded())
    {
	#ifdef _DEBUG
		std::cout << "Added rate population '"<< population->getName()<<"' on place " << mean_populations_.size()<<std::endl;
	#endif
		mean_populations_.push_back(static_cast<MeanPopulation*>(population));
    }
    else
    {
	#ifdef _DEBUG
		std::cout << "Added spike population '"<< population->getName()<<"' on place " << spike_populations_.size()<<std::endl;
	#endif
        spike_populations_.push_back(static_cast<SpikePopulation*>(population));
    }
}

/*
void Network::connect(int prePopulationID, int postPopulationID, Connector *connector, int projectionID, int targetID) {
	connector->connect(populations_[prePopulationID], populations_[postPopulationID], projectionID, targetID);
}
*/

void Network::disconnect(int prePopulationID, int postPopulationID, int targetID) {
	if (targetID == -1)
		populations_[postPopulationID]->removeProjections(populations_[prePopulationID]);
	else
		populations_[postPopulationID]->removeProjection(populations_[prePopulationID], targetID);
}

void Network::run(int steps) {
#ifdef ANNAR_PROFILE
        std::cout << "Run simulation with "<< omp_get_max_threads() << " thread(s)." << std::endl;
#endif

    double start, stop;
    #pragma omp parallel
    {
        for(int i =0; i<steps; i++)
        {
        #ifdef _DEBUG
            #pragma omp master
            std::cout << "current step " << i << " ANNarchy "<< ANNarchy_Global::time << std::endl;
        #endif

        #ifdef ANNAR_PROFILE
            #pragma omp barrier
			#pragma omp master
            {
            start = omp_get_wtime();
            }
        #endif

            //
            // parallel population wise
            #pragma omp for
            for(int p=0; p<(int)spike_populations_.size(); p++)
            {
                spike_populations_[p]->prepareNeurons();
            }

            //
            // parallel neuron wise
            for(int p=0; p<(int)mean_populations_.size(); p++)
            {
                mean_populations_[p]->metaSum();
            }
            #pragma omp barrier

            //
            // parallel neuron wise
            for(int p=0; p<(int)mean_populations_.size(); p++)
            {
                mean_populations_[p]->metaStep();
            }
            #pragma omp barrier

            //
            // parallel neuron wise
            for(int p=0; p<(int)spike_populations_.size(); p++)
            {
                spike_populations_[p]->metaStep();
            }
            #pragma omp barrier

            //
            // parallel population wise
            #pragma omp for
            for(int p=0; p<(int)mean_populations_.size(); p++)
            {
                mean_populations_[p]->globalOperations();
            }

			#pragma omp for
			for(int p=0; p<(int)spike_populations_.size(); p++)
			{
				spike_populations_[p]->globalOperations();
			}

            //
            // parallel neuron wise
            for(int p=0; p<(int)mean_populations_.size(); p++)
            {
           		mean_populations_[p]->metaLearn();
            }
            #pragma omp barrier
            for(int p=0; p<(int)spike_populations_.size(); p++)
            {
           		spike_populations_[p]->metaLearn();
            }
            #pragma omp barrier

            //
            // parallel population wise
            #pragma omp for
            for(int p=0; p<(int)mean_populations_.size(); p++)
            {
                mean_populations_[p]->record();
            }
			#pragma omp for
			for(int p=0; p<(int)spike_populations_.size(); p++)
			{
				spike_populations_[p]->record();
			}

        #ifdef ANNAR_PROFILE
			#pragma omp barrier
			#pragma omp master
            {
                stop = omp_get_wtime();
                Profile::profileInstance()->appendTimeNet( (stop - start)*1000.0 );
            }
        #endif

            #pragma omp master
            {
                ANNarchy_Global::time++;
            }
        }
    }

}
