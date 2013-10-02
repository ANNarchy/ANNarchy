/*
 *    Projection.cpp
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
#include "Projection.h"

Projection::~Projection() {
#ifdef _DEBUG
    std::cout<<"Projection::Destructor"<<std::endl;
#endif
    
    if(!rank_.empty())
        rank_.erase(rank_.begin(), rank_.end());
    if(!value_.empty())
        value_.erase(value_.begin(), value_.end());
    if(!delay_.empty())
        delay_.erase(delay_.begin(), delay_.end());

}

void Projection::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
	rank_ = rank;
	value_ = value;
	delay_ = delay;

	constDelay_ = true;
	maxDelay_ =0;

	if(!delay_.empty()) {
		maxDelay_=delay[0];

		for(auto it=delay.begin(); it!=delay.end();it++) {
			if(*it>maxDelay_)
				maxDelay_ = *it;

			if(*it != maxDelay_)
				constDelay_ = false;
		}

		//will be set by derived projection!!!
		//pre_population_->setMaxDelay(maxDelay_);
	}
}

int Projection::addSynapse(int rank, DATA_TYPE value, int delay) {
    bool found = false;
    for(unsigned int i=0; i < rank_.size(); i++) {
        if(rank_[i] == rank ) {
            found = true;
            continue;
        }
    }

    if(!found){
        rank_.push_back(rank);
        value_.push_back(value);
        delay_.push_back(delay);
        return 0;
    }else{
        return -1;
    }
}

int Projection::removeSynapse(int rank) {
#ifdef _DEBUG
    std::cout << "suppress synapse - pre = "<<rank<<std::endl;
    std::cout << "check "<<rank_.size()<<" synapses."<< std::endl;
#endif

    for(unsigned int i=0; i < rank_.size(); i++) {
        if(rank_[i] == rank ) {
           std::cout << "found the synapse at: "<< i <<std::endl;
           rank_.erase(rank_.begin()+i);
           value_.erase(value_.begin()+i);

           if (delay_.size() > 1)
               delay_.erase(delay_.begin()+i);

           return 0;
        }
    }

    return -1;
}
