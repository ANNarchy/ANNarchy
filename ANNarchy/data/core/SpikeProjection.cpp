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

SpikeProjection::SpikeProjection(std::string pre, std::string post, int target): Projection()
{
#ifdef _DEBUG
	std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << "', target = '" << target << "', coding = 'spike' ) " << std::endl;
#endif
	pre_population_ = static_cast<class SpikePopulation*>(Network::instance()->getPopulation(pre));
	post_population_ = static_cast<class SpikePopulation*>(Network::instance()->getPopulation(post));

	target_ = target;
	post_population_->addProjection(this);

	nbDendrites_ = static_cast<int>(post_population_->getNeuronCount());
	dendrites_ = std::vector< class SpikeDendrite* >(nbDendrites_, NULL);
}

void SpikeProjection::globalLearn()
{
	#pragma omp for
	for ( int d = 0; d < nbDendrites_; d++ )
	{
		dendrites_[d]->globalLearn();
	}

	#pragma omp barrier
}

void SpikeProjection::localLearn()
{
#ifdef _DEBUG
	std::cout << "number of dendrites: " << nbDendrites_ << std::endl;
#endif

	#pragma omp for
	for ( int d = 0; d < nbDendrites_; d++ )
	{
		dendrites_[d]->localLearn();
	}

	#pragma omp barrier
}

void SpikeProjection::postEvent(std::vector<int> post_ranks)
{
#ifdef _DEBUG
	std::cout << "number of post events: " << post_ranks.size() << std::endl;
#endif

	for ( int r = 0; r < post_ranks.size(); r++ )
	{
		dendrites_[r]->postEvent();
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
			dendrites_[postNeuronRank] = static_cast<SpikeDendrite*>(dendrite);
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

}

void SpikeProjection::record()
{
	for ( auto it = dendrites_.begin(); it != dendrites_.end(); it++ )
	{
		(*it)->record();
	}
}
