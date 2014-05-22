/*
 *    SpikeProjection.cpp
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
#include "SpikeProjection.h"
#include "SpikePopulation.h"
#include "SpikeDendrite.h"

SpikeProjection::SpikeProjection(): Projection()
{

}

void SpikeProjection::globalLearn()
{
	#pragma omp for
	for ( int d = 0; d < nbDendrites_; d++ )
	{
		static_cast<SpikeDendrite*>(dendrites_[d])->globalLearn();
	}

	#pragma omp barrier
}

void SpikeProjection::localLearn()
{
#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
	std::cout << "LocalLearn: number of dendrites: " << nbDendrites_ << std::endl;
#endif

	#pragma omp for
	for ( int n = 0; n < nbDendrites_; n++ )
	{
		if ( !dendrites_[n] )
			continue;

		static_cast<SpikeDendrite*>(dendrites_[n])->localLearn();
	}

	#pragma omp barrier
}

void SpikeProjection::postEvent(std::vector<int> post_ranks)
{
#ifdef _DEBUG
	std::cout << "number of post events: " << post_ranks.size() << std::endl;
#endif

	for ( unsigned int n = 0; n < post_ranks.size(); n++ )
	{
		if ( !dendrites_[n] )
			continue;

		static_cast<SpikeDendrite*>(dendrites_[n])->postEvent();
	}
}

Population* SpikeProjection::getPrePopulation()
{
	return static_cast<class Population*>(pre_population_);
}

void SpikeProjection::addDendrite(int postNeuronRank, class Dendrite *dendrite)
{
	if ( postNeuronRank < nbDendrites_ )
	{
	#ifdef _DEBUG
		std::cout << "Projection ( ptr = " << this << " ): added dendrite ( ptr = " << dendrite << " ) to neuron " << postNeuronRank << std::endl;
	#endif
		if ( dendrites_[postNeuronRank] == NULL)
		{
			dendrites_[postNeuronRank] = dendrite;
		}
		else
		{
		#ifdef _DEBUG
			std::cout << "Warning: already attached a dendrite ( ptr = " << dendrites_[postNeuronRank] << " ) to neuron " << postNeuronRank << std::endl;
		#endif
		}
	}
	else
	{
		std::cout << "Error on attaching dendrite to neuron " << postNeuronRank << ", expected a rank < " << dendrites_.size() << std::endl;
	}

}

Dendrite *SpikeProjection::getDendrite(int postNeuronRank)
{
	return dendrites_[postNeuronRank];
}

void SpikeProjection::removeDendrite(int postNeuronRank, class Population *pre)
{
	std::cout << "Need to implemented in CPP core."<< std::endl;
}

void SpikeProjection::record()
{
	for ( auto it = dendrites_.begin(); it != dendrites_.end(); it++ )
	{
		if (*it == NULL)
			continue;

		static_cast<SpikeDendrite*>((*it))->record();
	}
}

void SpikeProjection::initValues(int postNeuronRank)
{
	if (dendrites_[postNeuronRank] != NULL)
		static_cast<SpikeDendrite*>(dendrites_[postNeuronRank])->initValues();
}

int SpikeProjection::nbSynapses(int post_rank)
{
	if ( !dendrites_[post_rank] )
		return 0;
	else
		return dendrites_[post_rank]->getSynapseCount();
}
