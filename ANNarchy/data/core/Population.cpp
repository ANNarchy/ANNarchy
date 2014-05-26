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
#include "Global.h"
#include <exception>
#include <typeinfo>
#include "RateProjection.h"

Population::Population(std::string name, unsigned int nbNeurons, bool isRateType)
{
    name_ = name;
    nbNeurons_ = nbNeurons;
    dt_ = 1.0;
    maxDelay_ = 0;
    isRateType_ = isRateType;

    projections_.clear();
    typedProjections_ = std::vector< std::vector<Projection*> >();

#ifdef ANNAR_PROFILE
    try
    {
        Profile::profileInstance()->addLayer(name_);
    }
    catch(std::exception e)
    {
        std::cout << "Can't attach population to profile instance." << std::endl;
        std::cout << e.what() << std::endl;
    }
#endif
}

void Population::addProjection(Projection* projection)
{
#ifdef _DEBUG
	bool isRateCoded = projection->isRateCoded();
	std::string tmp = isRateCoded ? "rate" : "spike";
    std::cout << name_ << ": added projection ( ptr = "<< projection << ", " << tmp << " coded, target = "<< projection->getTarget() << ")" <<std::endl;
#endif
	projections_.push_back(projection);

	if( projection->getTarget() >= typedProjections_.size() )
	{
	#ifdef _DEBGU
		std::cout << "extend typed projection by "<< projection->getTarget() << std::endl;
	#endif
		typedProjections_.resize( projection->getTarget()+1, std::vector<class Projection*>() );
	}

	typedProjections_[projection->getTarget()].push_back(projection);
}

void Population::removeProjections(Population* pre)
{
#ifdef _DEBUG
    std::cout << name_ << ": remove projections to '"<< pre->getName() << "'" <<std::endl;
#endif

}

void Population::removeProjection(Population* pre, int target)
{
#ifdef _DEBUG
    std::cout << name_ << ": remove projection to '"<< pre->getName() << "', target = " << target << "'" <<std::endl;
#endif

}

class Projection* Population::getProjection(class Population* pre, int target)
{
	for(auto it = projections_.begin(); it != projections_.end(); it++ )
	{
		if ( ((*it)->getPrePopulation() == pre) && ( (*it)->getTarget() == target) )
			return *it;
	}

	return NULL;
}

