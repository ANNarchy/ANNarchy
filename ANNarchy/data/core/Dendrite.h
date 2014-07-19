/*
 *    Dendrite.h
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
#ifndef __ANNARCHY_DENDRITE_H__
#define __ANNARCHY_DENDRITE_H__

#include "Global.h"

/**
 * 	\brief		basic description of a dendrite in ANNarchy
 * 	\details	extended by RateDendrite and SpikeDendrite.
 */
class Dendrite
{
public:
	/**
	 * 	\brief		Constructor.
	 * 	\param[in]	rateCoded	set to true if the population is rate coded.
	 * 	\param[in]	proj		related projection
	 */
	Dendrite(bool rateCoded, class Projection* proj);

	/**
	 * 	\brief		Destructor.
	 * 	\details	major implemented by children
	 */
    virtual ~Dendrite();

    /**
     *  \brief		is the current dendrite rate coded.
     *  \details	the variable isRateCoded__ is set by the inheriting class.
     */
    bool isRateCoded() { return isRateCoded_; }

    /**
     * 	\brief		returns the reference to presynaptic populations
     * 	\details	is abstract, cause the reference is hold by child class.
     */
	virtual class Population* getPrePopulation() = 0;

    /**
     * 	\brief		add a synapse to this dendrite
     * 	\details	is abstract, cause the behavior is implemented by the child class.
     * 	\param[in]	rank	presynaptic rank
     * 	\param[in]	w    	synaptic weight
     * 	\param[in]	delay	synaptic delay
     */
	virtual int addSynapse(int rank, DATA_TYPE w, int delay) = 0;

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
    void setDt(DATA_TYPE dt) { dt_ = dt; }

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
	std::vector<int> get_delay() { return delay_; };

	/**
	 * 	\brief		get maximum value of synaptic delays
	 * 	\return		maximum synaptic delay
	 */
	int get_max_delay() { return maxDelay_; }

	/**
	 * 	\brief		set synaptic delays
	 * 	\details	automatically update constDelay_ and maxDelay_.
	 * 	\param[in]	vector of synaptic delays.
	 */
	virtual void set_delay(std::vector<int> delay);

	/**
	 * 	\brief		get ranks of presynaptic neurons
	 * 	\return		ranks of presynaptic neurons
	 */
	std::vector<int> get_rank() { return rank_; }

	/**
	 * 	\brief		set ranks of presynaptic neurons
	 * 	\details	if the projection is spike coded, the inverted rank arrays is updated.
	 * 	\param[in]	rank	ranks of presynaptic neurons
	 */
	void set_rank(std::vector<int> rank);

	/**
	 * 	\brief		get synaptic weights
	 */
	std::vector<DATA_TYPE> get_w();

	/**
	 * 	\brief		set synaptic weights
	 */
	void set_w(std::vector<DATA_TYPE> value);

    DATA_TYPE get_single_w(int rank) { return this->w_[rank]; }

    void set_single_w(int rank, DATA_TYPE value) { this->w_[rank] = value; }

    /**
     * 	\brief		return if the projection is between rate coded or spike coded populations
     */
    bool isMeanRateCoded() { return isRateCoded_; }

    /**
	 * 	\brief		determine if the synapses learn at current time step
	 * 	\details	only True if learnable_ is True and time % learn frequency is equal to the learn offset, called only internally by Population::metaLearn()
	 */
    bool isLearning();

    /**
	 * 	\brief		returns learnable property
	 * 	\details	called by cython wrapper.
	 */
    bool isLearnable();

    /**
	 * 	\brief		set learn frequency
	 * 	\details	called by cython wrapper.
	 */
    int getLearnFrequency();

    /**
     * 	\brief		get learn offset
	 * 	\details	called by cython wrapper.
     */
    int getLearnOffset();

protected:
    class Projection *proj_;

    int post_neuron_rank_; 	///< neuron where this dendrite is attached to
    int target_;	///< dendrite type
    int nbSynapses_;	///< number of synapses within this dendrite
    DATA_TYPE dt_;	///< discretization constant

    std::vector<int> rank_; ///< pre-ranks for connection post->pre
    //std::map<int, int> inv_rank_; ///< inversed access ranks, to ensure, that a presynaptic neurons accesses the right values

    std::vector<int> delay_;	///< synaptic delay
    bool constDelay_;	///< true: a delay != 0 and common to all synapses
    int maxDelay_;	///< maximum delay value in this dendrite

    std::vector<DATA_TYPE> w_;	///< synaptic weights

    bool isRateCoded_;
};
#endif

