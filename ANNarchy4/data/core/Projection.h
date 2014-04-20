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
	Projection(bool rateCoded);

    virtual ~Projection();

    /**
     * 	\brief		returns the reference to presynaptic populations
     * 	\details	is abstract, cause the reference is hold by child class.
     */
	virtual class Population* getPrePopulation() = 0;

    /**
     * 	\brief		add a synapse to this dendrite
     * 	\details	is abstract, cause the behavior is implemented by the child class.
     * 	\param[in]	rank	presynaptic rank
     * 	\param[in]	value	synaptic weight
     * 	\param[in]	delay	synaptic delay
     */
	virtual int addSynapse(int rank, DATA_TYPE value, int delay) = 0;

    /**
     * 	\brief		removes a synapse from this dendrite
     * 	\details	is abstract, cause the behavior is implemented by the child class.
     * 	\param[in]	rank	presynaptic rank, identifying the correct synapse
     */
    virtual int removeSynapse(int rank) = 0;

    /**
     * 	\brief		removes all synapses from this dendrite
     * 	\details	is abstract, cause the behavior is implemented by the child class.
     */
    virtual int removeAllSynapses() = 0;

    /**
     * 	\brief		record synaptic data
     * 	\details	is abstract, cause the behavior is implemented by the child class.
     */
    virtual void record() = 0;

    /**
     * 	\brief		get discretization time constant
     * 	\details	common to all objects, its globally changed through python.
     * 	\return		value of dt
     */
    DATA_TYPE getDt() { return dt_; }

    /**
     * 	\brief		set discretization time constant
     * 	\details	common to all objects, its globally changed through python.
     * 	\param[in]	dt 	new value of dt
     */
    void setDt(float dt) { dt_ = dt; }

	/**
	 * 	\brief		get number of synapses
	 * 	\return		number of synapses
	 */
	int getSynapseCount() { return nbSynapses_; }

	/**
	 * 	\brief		get projection target
	 * 	\return		integer id of projection target
	 */
	int getTarget() { return target_; }

	/**
	 * 	\brief		get synaptic delays
	 * 	\return		synaptic delays
	 */
	std::vector<int> getDelay() { return delay_; };

	/**
	 * 	\brief		set synaptic delays
	 * 	\details	automatically update constDelay_ and maxDelay_.
	 * 	\param[in]	vector of synaptic delays.
	 */
	void setDelay(std::vector<int> delay);

	/**
	 * 	\brief		get ranks of presynaptic neurons
	 * 	\return		ranks of presynaptic neurons
	 */
	std::vector<int> getRank() { return rank_; }

	/**
	 * 	\brief		set ranks of presynaptic neurons
	 * 	\details	if the projection is spike coded, the inverted rank arrays is updated.
	 * 	\param[in]	rank	ranks of presynaptic neurons
	 */
	void setRank(std::vector<int> rank);

	/**
	 * 	\brief		get synaptic weights
	 */
	std::vector<DATA_TYPE> getValue() { return value_; }

	/**
	 * 	\brief		set synaptic weights
	 */
	void setValue(std::vector<DATA_TYPE> value) { value_ = value; }

    /**
     * 	\brief		return if the projection is between rate coded or spike coded populations
     */
    bool isMeanRateCoded() { return isRateCoded_; }

    /**
	 * 	\brief		determine if the synapses learn at current time step
	 * 	\details	only True if learnable_ is True and time % learn frequency is equal to the learn offset, called only internally by Population::metaLearn()
	 */
    bool isLearning() { return (learnable_ && ((ANNarchy_Global::time)%learnFrequency_ == learnOffset_)); }

    /**
	 * 	\brief		set learnable property
	 * 	\details	called by cython wrapper.
	 */
    void setLearnable( bool learnable ) { learnable_ = learnable; }

    /**
	 * 	\brief		returns learnable property
	 * 	\details	called by cython wrapper.
	 */
    bool isLearnable() { return learnable_; }

    /**
	 * 	\brief		set learn frequency
	 * 	\details	called by cython wrapper.
	 */
    void setLearnFrequency(int learnFrequency) { learnFrequency_ = learnFrequency; }

    /**
	 * 	\brief		set learn frequency
	 * 	\details	called by cython wrapper.
	 */
    int getLearnFrequency() { return learnFrequency_; }

    /**
     * 	\brief		set learn offset
	 * 	\details	called by cython wrapper.
     */
    void setLearnOffset(int learnOffset) { learnOffset_ = learnOffset; }

    /**
     * 	\brief		get learn offset
	 * 	\details	called by cython wrapper.
     */
    int getLearnOffset() { return learnOffset_; }

protected:
    int post_neuron_rank_; 	///< neuron where this dendrite is attached to
    int target_;	///< projection type
    int nbSynapses_;	///< number of synapses
    DATA_TYPE dt_;	///< discretization constant

    std::vector<int> rank_; ///< pre-ranks for connection post->pre
    std::map<int, int> inv_rank_; ///< pre-ranks of synapses for connection pre->post (needed by spikes)

    std::vector<int> delay_;	///< synaptic delay
    bool constDelay_;	///< true: a delay != 0 and common to all synapses
    int maxDelay_;	///< maximum delay value in this dendrite

    std::vector<DATA_TYPE> value_;	///< synaptic weights

    int learnFrequency_; 	///< the learn frequency determines after which amount of steps the next learn will executed.
    int learnOffset_;	///< the learn offset determines the time step within the learn frequency window, where learning will be executed.
    bool learnable_;
    bool isRateCoded_;
};
#endif

/*
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

	virtual int addSynapse(int rank, DATA_TYPE value, int delay);

    virtual int removeSynapse(int rank);

    virtual bool isPreSynaptic(class Population *pop) {}

    virtual void preEvent(int rank) {}

    virtual void postEvent() {}

    virtual void invertRanks() { }

    virtual void record() { }

    void setLearnable( bool learnable ) { learnable_ = learnable; }

    bool isLearnable() { return learnable_; }

    bool isLearning() { return (learnable_ && ((time_)%learnFrequency_ == learnOffset_)); }

    void setLearnFrequency(int learnFrequency) { learnFrequency_ = learnFrequency; }

    int getLearnFrequency() { return learnFrequency_; }

    void setLearnOffset(int learnOffset) { learnOffset_ = learnOffset; }

    int getLearnOffset() { return learnOffset_; }

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

    int learnFrequency_;	// amount of timesteps till next learning (default 1)
    int learnOffset_;	// at which point in the frequence we learn (default 0)
    bool learnable_;
    bool constDelay_;
    int maxDelay_;
    int time_;
};
 */
