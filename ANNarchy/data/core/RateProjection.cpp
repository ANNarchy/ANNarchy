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

RateProjection::RateProjection(std::string pre, std::string post, int target): Projection()
{
#ifdef _DEBUG
	std::cout << "Establish projection ( ptr = "<< this <<") between pre = '"<< pre << "', post ='"<< post << ", target = "<< target << std::endl;
#endif
	pre_population_ = static_cast<class RatePopulation*>(Network::instance()->getPopulation(pre));
	post_population_ = static_cast<class RatePopulation*>(Network::instance()->getPopulation(post));

	post_population_->addProjection(this);
	dendrites_ = std::vector< std::vector<RateDendrite*> >(post_population_->getNeuronCount(), std::vector<RateDendrite*>());

	target_ = target;
	isRateCoded_ = true;
}

Population* RateProjection::getPrePopulation()
{
	return static_cast<class Population*>(pre_population_);
}

void RateProjection::addDendrite(int postNeuronRank, class Dendrite *dendrite)
{
#ifdef _DEBUG
    std::cout << "Projection ( ptr = " << this << " ): added dendrite ( ptr = " << dendrite << ") to neuron " << postNeuronRank << std::endl;
#endif
	try
	{
		dendrites_.at(postNeuronRank).push_back(static_cast<RateDendrite*>(dendrite));
	}
	catch (std::exception &e)
	{
		std::cout << std::endl;
		std::cout << "Caught: " << e.what() << std::endl;
		std::cout << "caused by: attach a dendrite to neuron " << postNeuronRank <<" but there only " << post_population_->getNeuronCount() << " neurons" << std::endl;
		std::cout << std::endl;
	};
}

void RateProjection::computeSum()
{
#ifdef _DEBUG
	std::cout << "number of dendrites:" << dendrites_.size() << std::endl;
#endif

	for ( unsigned int n = 0; n != dendrites_.size(); n++ )
	{
		for ( auto d_it = dendrites_[n].begin(); d_it != dendrites_[n].end(); d_it++ )
		{
			std::cout << "dendrite(" << n << "): " << (*d_it)->getSynapseCount()<< std::endl;
			(*d_it)->computeSum();
		}
	}
}
