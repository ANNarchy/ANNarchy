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

#include "RatePopulation.h"
#include "SpikePopulation.h"
#include "ANNarchy.h"
#include "Global.h"
#include "Includes.h"

Network::Network()
{
	rate_populations_.clear();
	spike_populations_.clear();

	ANNarchy_Global::time = 0;
}

Network::~Network()
{
    std::cout << "Network destructor." << std::endl;

	while(!rate_populations_.empty())
	{
		delete rate_populations_.back();
		rate_populations_.pop_back();
	}

	while(!spike_populations_.empty())
	{
		delete spike_populations_.back();
		spike_populations_.pop_back();
	}
}

class Population* Network::getPopulation(std::string name)
{
	for(auto it = rate_populations_.cbegin(); it != rate_populations_.cend(); it++)
	{
		if((*it)->getName().compare(name)==0)
			return (*it);
	}

	for(auto it = spike_populations_.cbegin(); it != spike_populations_.cend(); it++)
	{
		if((*it)->getName().compare(name)==0)
			return (*it);
	}

#ifdef _DEBUG
	std::cout << "Population name = "<< name <<" not exist."<<std::endl;
#endif
	return NULL;
}

class Population* Network::getPopulation(std::string name, bool isRateCoded)
{
	if (isRateCoded)
	{
		for(auto it = rate_populations_.cbegin(); it != rate_populations_.cend(); it++)
		{
			if((*it)->getName().compare(name)==0)
				return (*it);
		}
	}
	else
	{
		for(auto it = spike_populations_.cbegin(); it != spike_populations_.cend(); it++)
		{
			if((*it)->getName().compare(name)==0)
				return (*it);
		}
	}

#ifdef _DEBUG
	std::cout << "Population name = "<< name <<" not exist."<<std::endl;
#endif
	return NULL;
}

class Population* Network::getPopulation(unsigned int id)
{

	for(auto it = rate_populations_.cbegin(); it != rate_populations_.cend(); it++)
	{
		if ((*it)->getRank() == id)
			return (*it);
	}

	for(auto it = spike_populations_.cbegin(); it != spike_populations_.cend(); it++)
	{
		if ((*it)->getRank() == id)
			return (*it);
	}

	return NULL;
}

class Population* Network::getPopulation(unsigned int id, bool isRateCoded)
{

	if ( isRateCoded && id < rate_populations_.size() )
	{
		return rate_populations_[id];
	}
	else if ( !isRateCoded && id < spike_populations_.size() )
	{
		return spike_populations_[id];
	}
	else
	{
	#ifdef _DEBUG
		std::string tmp = isRateCoded ? "true" : "false";
		std::cout << "Population id = "<<id<<", rate = "<< tmp << " not exist" <<std::endl;
	#endif
		return NULL;
	}
}

void Network::addPopulation(class Population* population)
{
    if(population->isMeanRateCoded())
    {
	#ifdef _DEBUG
		std::cout << "Added rate population '"<< population->getName()<<"' on place " << rate_populations_.size()<<std::endl;
	#endif
		rate_populations_.push_back(static_cast<RatePopulation*>(population));
    }
    else
    {
	#ifdef _DEBUG
		std::cout << "Added spike population '"<< population->getName()<<"' on place " << spike_populations_.size()<<std::endl;
	#endif
        spike_populations_.push_back(static_cast<SpikePopulation*>(population));
    }
}

void Network::connect(int prePopulationID, int postPopulationID, int projectionID, int targetID, bool spike, std::string filename)
{
	std::fstream file(filename, std::ios_base::in);
	if (!file.is_open()) {
		std::cout << "Failed to open file '"<< filename <<"' ... " << std::endl;
		return;
	}

	std::cout << "Create pattern from file: "<< filename << std::endl;
	std::string line;
	int preNeuronRank = -1;
	int previousRank = -1; // to determine start of new dendrite
	int postNeuronRank = -1;
	DATA_TYPE value = 0.0;
	int delay = 0;
	Dendrite* dendrite = NULL;

	int line_counter=0;
	int dendrite_counter = 0;

	auto pre = getPopulation(prePopulationID);
	auto post = getPopulation(postPopulationID);

	while(getline(file, line))
	{
		std::vector<std::string> elem = ANNarchy_Global::split(line, ',');

		preNeuronRank = stoi(elem[0]);
		postNeuronRank = stoi(elem[1]);
		value = (DATA_TYPE)(stod(elem[2]));
		delay = stoi(elem[3]);

		if (postNeuronRank != previousRank)
		{
			dendrite = createProjInstance().getInstanceOf(projectionID, pre, post, postNeuronRank, targetID);
			dendrite->removeAllSynapses();	// just for the case there are some previously allocated datas
			previousRank = postNeuronRank;

			dendrite_counter++;
		}
	#ifdef _DEBUG
		std::cout << "add synapse: pre = "<< preNeuronRank << ", post =" << postNeuronRank<< ", value = "<< value << ", delay = "<< delay << std::endl;
	#endif
		dendrite->addSynapse(preNeuronRank, value, delay);
		line_counter++;
	}

	std::cout << "read "<< line_counter << " line(s) and created "<< dendrite_counter <<" dendrites"<< std::endl;
}

void Network::disconnect(int prePopulationID, int postPopulationID, bool preIsSpike, bool postIsSpike, int targetID)
{
	auto pre = getPopulation(prePopulationID, preIsSpike);
	auto post = getPopulation(postPopulationID, postIsSpike);

	if ( pre && post )
	{
		if (targetID == -1)
			post->removeDendrites(pre);
		else
			post->removeDendrite(pre, targetID);
	}
	else
	{
	#ifdef _DEBUG
		std::cout << "Projection does not exist ... " << std::endl;
	#endif
	}
}

void Network::run(int steps)
{
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
            for(unsigned int p = 0; p < spike_populations_.size(); p++)
            {
                spike_populations_[p]->prepareNeurons();
            }

            //
            // parallel neuron wise
            for(unsigned int p = 0; p < rate_populations_.size(); p++)
            {
                rate_populations_[p]->metaSum();
            }
            #pragma omp barrier

            //
            // parallel neuron wise
            for(unsigned int p = 0; p < rate_populations_.size(); p++)
            {
            	rate_populations_[p]->metaStep();
            }
            #pragma omp barrier

            //
            // parallel neuron wise
            for(unsigned int p = 0; p < spike_populations_.size(); p++)
            {
                spike_populations_[p]->metaStep();
            }
            #pragma omp barrier

            //
            // parallel population wise
            #pragma omp for
            for(unsigned int p = 0; p < rate_populations_.size(); p++)
            {
            	rate_populations_[p]->globalOperations();
            }

			#pragma omp for
			for(unsigned int p = 0; p < spike_populations_.size(); p++)
			{
				spike_populations_[p]->globalOperations();
			}

            //
            // parallel neuron wise
            for(int p = 0; p < rate_populations_.size(); p++)
            {
            	rate_populations_[p]->metaLearn();
            }
            #pragma omp barrier

            for(int p = 0; p < spike_populations_.size(); p++)
            {
           		spike_populations_[p]->metaLearn();
            }
            #pragma omp barrier

            //
            // parallel population wise
            #pragma omp for
            for(unsigned int p = 0; p < rate_populations_.size(); p++)
            {
            	rate_populations_[p]->record();
            }

			#pragma omp for
			for(unsigned int p = 0; p < spike_populations_.size(); p++)
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
