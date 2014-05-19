/*
 *    RateProjection.cpp
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
#include "RateProjection.h"
#include "RatePopulation.h"
#include "RateDendrite.h"

RateProjection::RateProjection(): Projection()
{

}

void RateProjection::computeSum()
{
#ifdef _DEBUG
	std::cout << "number of dendrites: " << nbDendrites_ << std::endl;
#endif

	#pragma omp for
	for ( int n = 0; n < nbDendrites_; n++ )
	{
		if (!dendrites_[n])
			continue;

	#ifdef _DEBUG
		std::cout << "dendrite( ptr = " << dendrites_[n] << ", n = " << n << "): " << dendrites_[n]->getSynapseCount() << " synapse(s) " << std::endl;
	#endif
		dendrites_[n]->computeSum();
	}
}

void RateProjection::globalLearn()
{
#ifdef _DEBUG
	#pragma omp master
	{
		std::cout << "GlobalLearn: number of dendrites: " << nbDendrites_ << std::endl;
	}
#endif

	if ( ANNarchy_Global::time % learnFrequency_ == learnOffset_ )
	{
		#pragma omp for
		for ( int n = 0; n < nbDendrites_; n++ )
		{
			if (!dendrites_[n])
				continue;

			dendrites_[n]->globalLearn();
		}
	}
}

void RateProjection::localLearn()
{
#ifdef _DEBUG
	#pragma omp master
	{
	std::cout << "LocalLearn: number of dendrites: " << nbDendrites_ << std::endl;
	}
#endif

	if ( ANNarchy_Global::time % learnFrequency_ == learnOffset_ )
	{
		#pragma omp for
		for ( int n = 0; n < nbDendrites_; n++ )
		{
			if (!dendrites_[n])
				continue;

			dendrites_[n]->localLearn();
		}
	}
}

DATA_TYPE RateProjection::getSum(int neuron)
{
	if ( neuron >= dendrites_.size() )
	{
		std::cout << "No dendrite " << neuron << "on this projection."<< std::endl;
		return 0.0;
	}
	else if  ( !dendrites_[neuron] )
	{
		return 0.0;
	}
	else
	{
		return dendrites_[neuron]->getSum();
	}
}

void RateProjection::addDendrite(int postNeuronRank, class Dendrite *dendrite)
{
	if ( postNeuronRank < nbDendrites_ )
	{
	#ifdef _DEBUG
		std::cout << "Projection ( ptr = " << this << " ): added dendrite ( ptr = " << dendrite << " ) to neuron " << postNeuronRank << std::endl;
	#endif
		if ( dendrites_[postNeuronRank] == NULL)
		{
			dendrites_[postNeuronRank] = static_cast<RateDendrite*>(dendrite);
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

Dendrite *RateProjection::getDendrite(int postNeuronRank)
{
	return dendrites_[postNeuronRank];
}

int RateProjection::nbSynapses(int post_rank) { 
	return dendrites_[post_rank]->getSynapseCount();
}

void RateProjection::removeDendrite(int postNeuronRank, class Population *pre)
{
	std::cout << "Need to implemented in CPP core."<< std::endl;
}

void RateProjection::addSynapse(int postNeuronRank, int pre, double value, int delay)
{
#ifdef _DEBUG
	std::cout << "Dendrite "<<postNeuronRank << " extend by synapse ( pre = " << pre << ", value = "<< value << ", delay = "<< delay << ")." << std::endl;
#endif

	if (dendrites_[postNeuronRank] != NULL)
		dendrites_[postNeuronRank]->addSynapse(pre, value, delay);
}

void RateProjection::removeSynapse(int postNeuronRank, int pre)
{
#ifdef _DEBUG
	std::cout << "Dendrite "<< postNeuronRank << " remove synapse to pre = " << pre << "." << std::endl;
#endif

	if (dendrites_[postNeuronRank] != NULL)
		dendrites_[postNeuronRank]->removeSynapse(pre);
}

void RateProjection::initValues(int postNeuronRank)
{
	if (dendrites_[postNeuronRank] != NULL)
		dendrites_[postNeuronRank]->initValues();
}

void RateProjection::record()
{
	for ( int n = 0; n < nbDendrites_; n++ )
	{
		if ( !dendrites_[n] )
			continue;

		dendrites_[n]->record();
	}
}

