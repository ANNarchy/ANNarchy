/*
 *    Dendrite.cpp
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
#include "Dendrite.h"

Dendrite::Dendrite(bool rateCoded, class Projection* proj)
{
	isRateCoded_ = rateCoded;
	proj_ = proj;

    constDelay_ = true;
    maxDelay_ = 0;
    nbSynapses_ = 0;
}

Dendrite::~Dendrite()
{
#ifdef _DEBUG
    std::cout<<"Projection::Destructor (" << this << ")"<<std::endl;
#endif

    if(!rank_.empty())
        rank_.erase(rank_.begin(), rank_.end());
    if(!w_.empty())
        w_.erase(w_.begin(), w_.end());
    if(!delay_.empty())
        delay_.erase(delay_.begin(), delay_.end());

}

void Dendrite::set_delay(std::vector<int> delay)
{
#ifdef _DEBUG_DENDRITE_DATA
	std::cout << "Dendrite::setDelay (ptr = "<< this << ")"<< std::endl;
	for (auto it = delay.begin(); it != delay.end(); it++)
		std::cout << *it << " ";
	std::cout << std::endl;
#endif
#ifdef _DEBUG_DELAY
	std::cout << "Dendrite (ptr = " << this << ")" << std::endl;
	std::cout << "OLD: maxDelay = " << maxDelay_ << " and constDelay_ " << constDelay_ << std::endl;
#endif
	for(auto it=delay.begin(); it!=delay.end();it++)
            {
		if(*it>maxDelay_)
			maxDelay_ = *it;

		if(*it != maxDelay_)
			constDelay_ = false;
	}

#ifdef _DEBUG_DELAY
	std::cout << "NEW: maxDelay = " << maxDelay_ << " and constDelay_ " << constDelay_ << std::endl;
#endif
	delay_ = delay;
};

void Dendrite::set_rank(std::vector<int> rank)
{
#ifdef _DEBUG_DENDRITE_DATA
	std::cout << "Dendrite::setRank (ptr = "<< this << ")"<< std::endl;
	for (auto it = rank.begin(); it != rank.end(); it++)
		std::cout << *it << " ";
	std::cout << std::endl;
#endif
	rank_ = rank;
	nbSynapses_ = rank.size();

	if ( !isRateCoded_ )
	{
		inv_rank_.clear();
		for(int i = 0; i < rank_.size(); i++)
		{
		    auto tmp = std::pair<int,int>(rank_[i], i);
		    inv_rank_.insert( tmp );
		}
	}
}

std::vector<DATA_TYPE> Dendrite::get_w()
{
#ifdef _DEBUG_DENDRITE_DATA
	std::cout << "Dendrite::getW (ptr = "<< this << ")"<< std::endl;
	for (auto it = w_.begin(); it != w_.end(); it++)
		std::cout << *it << " ";
	std::cout << std::endl;
#endif
	return w_;
}

void Dendrite::set_w(std::vector<DATA_TYPE> value)
{
#ifdef _DEBUG_DENDRITE_DATA
	std::cout << "Dendrite::setW (ptr = "<< this << ")"<< std::endl;
	for (auto it = value.begin(); it != value.end(); it++)
		std::cout << *it << " ";
	std::cout << std::endl;
#endif
	w_ = value;
}

bool Dendrite::isLearning()
{
	return proj_->isLearning();
}

bool Dendrite::isLearnable()
{
	return proj_->isLearnable();
}

int Dendrite::getLearnFrequency()
{
	return proj_->getLearnFrequency();
}

int Dendrite::getLearnOffset()
{
	return proj_->getLearnOffset();
}


