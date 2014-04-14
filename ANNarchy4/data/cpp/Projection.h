/*
 *    Projection.h
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
#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include "Global.h"

class Projection{
public:
	Projection();

        virtual ~Projection();

	virtual void initValues();

	virtual void computeSum() {};

	virtual void globalLearn() {}

	virtual void localLearn() {}

	virtual class Population* getPrePopulation() = 0;

	DATA_TYPE getSum() { return sum_; }

	int getSynapseCount() { return rank_.size(); }

	int getTarget() { return target_; }

	std::vector<int> getDelay() { return delay_; };

	void setDelay(std::vector<int> delay) 
    {
        #ifdef _DEBUG
                std::cout << "OLD: maxDelay = " << maxDelay_ << " and constDelay_ " << constDelay_ << std::endl; 
        #endif
		for(auto it=delay.begin(); it!=delay.end();it++) 
                {
			if(*it>maxDelay_)
				maxDelay_ = *it;

			if(*it != maxDelay_)
				constDelay_ = false;
		}
        #ifdef _DEBUG
                std::cout << "NEW: maxDelay = " << maxDelay_ << " and constDelay_ " << constDelay_ << std::endl; 
        #endif
                delay_ = delay;
    };
    	
	std::vector<int> getRank() { return rank_; }

	void setRank(std::vector<int> rank) { rank_ = rank; nbWeights_ = rank.size(); }
    
	std::vector<DATA_TYPE> getValue() { return value_; }
	void setValue(std::vector<DATA_TYPE> value) { value_ = value; }
	
	DATA_TYPE getDt() { return dt_; }

	void setDt(DATA_TYPE dt) { dt_ = dt; }

	/**
	 *  \brief      Add synapse.
	 *  \param[IN]  rank    rank of the presynaptic neuron
	 *  \param[IN]  value   synaptic weight
	 *  \param[IN]  delay   delay
	 *  \return     error code: 0 (success), -1 (already existant)
	 */
	virtual int addSynapse(int rank, DATA_TYPE value, int delay);
	
	/**
	 *  \brief      Remove synapse.
	 *  \param[IN]  rank    rank of the presynaptic neuron
	 *  \return     error code: 0 (success), -1 (not existant)
	 */
    virtual int removeSynapse(int rank);
    
    virtual bool isPreSynaptic(class Population *pop) {}

    virtual void preEvent(int rank) {}

    virtual void postEvent() {}

    virtual void invertRanks() { }

    virtual void record() { }
protected:
    int post_neuron_rank_;
    int target_;
    int nbWeights_;

    std::vector<int> rank_; ///< pre ranks for connection post->pre
    std::map<int, int> inv_rank_; ///< pre-ranks of synapses

    std::vector<int> delay_;
    std::vector<DATA_TYPE> value_;

    DATA_TYPE sum_;

    DATA_TYPE post_rate_;
    std::vector<DATA_TYPE>* post_rates_;
    std::vector<DATA_TYPE>* pre_rates_;


    DATA_TYPE dt_;

    bool constDelay_;
    int maxDelay_;
    int time_;
};
#endif
