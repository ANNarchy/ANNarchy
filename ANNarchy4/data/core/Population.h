/*
 *    Population.h
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
#ifndef __ANNARCHY_POPULATION_H__
#define __ANNARCHY_POPULATION_H__

#include "Global.h"

/**
 * 	\brief		Basic definition of a population in ANNarchy
 * 	\details	inherited by MeanPopulation and SpikePopulation dependent on the chosen parallelization.
 * 				This class provides only simplest data common to all inherited classes.
 * 				Secondly this class provides a simple interface callable by Network.
 */
class Population
{
public:
	/**
	 * 	\brief		Initialize a population object.
	 * 	\details	Please note, that in no case this class is directly instantiated, always instantiate a child class depending on the parallelization method.
	 */
    Population( std::string name, unsigned int nbNeurons, bool isRateType );

    /**
     * 	\brief		Destructor
     * 	\details	No further logic implementd. The child class is COMPLETLY responsible for all the allocated data.
     * 				For easy use, you may use the removeProjections function in order to clean all the projection vectors.
     */
    virtual ~Population() { }

    /**
     * 	\brief		returns the name of population.
     * 	\details	within the CPP core this name is meaningless, it's might used by the wrappers.
     *	\return		name of population.
     */
    std::string getName() { return name_; }

    /**
     * 	\brief		returns the number of neurons.
     * 	\details	is not changed during runtime and set by constructor.
     *	\return		number of neurons
     */
    unsigned int getNeuronCount() { return nbNeurons_; }

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
     *  \brief		add a projection to a neuron
     *  \details	the given projection is attached to the postsynaptic neuron identified by given rank.
     *  \param[in]	neuron	postsynaptic rank of the neuron
     *  \param[in]	proj	projection instance to attach
     */
    void addDendrite(unsigned int neuron, class Dendrite* proj);

    /**
     *  \brief		get all projections of neuron n and a certain type
     *  \details	iterates over all assigned projections of neuron n and delete the ones with presynaptic population pre and the given projection type.
     *  \param[in]	pre		reference to the presynaptic populations
     *  \param[in]	type	integer id of the projection type
     */
    void removeDendrite(Population *pre, int type);

    /**
     *  \brief		get all projections of neuron n and a certain type
     *  \details	iterates over all assigned projections of neuron n and delete the ones with presynaptic population without attention to projection type.
     *  \param[in]	pre		reference to the presynaptic populations
     */
    void removeDendrites(Population *pre);

    /**
     *  \brief		get the projection of neuron n with a certain type and connected to a given presynaptic population
     *  \details	iterates over all assigned projections of neuron n and select the one projection
     *  			of given type and presynaptic population pre
     *  \param[in]	neuron	rank of the neuron
     *  \param[in]	type	integer id of the projection type
     *  \param[in]	pre		reference to the presynaptic population
     *  \return		instance of the searched Projection, if one exist in the current lists, otherwise NULL.
     */
    class Dendrite* getDendrite(unsigned int neuron, int type, Population* pre);

    /**
     *  \brief		get all projections of neuron n and a certain type
     *  \details	iterates over all assigned projections of neuron n and select the ones
     *  			of given type.
     *  \param[in]	neuron	rank of the neuron
     *  \param[in]	type	integer id of the projection type
     *  \return		std::vector of class Projection, if there some exists otherwise an empty vector is returned.
     */
    std::vector<class Dendrite*> getDendrites(unsigned int neuron, int type);

    /**
     *  \brief		is the current population rate coded.
     *  \details	the variable isRateType_ is set by the inheriting class.
     */
    bool isMeanRateCoded() { return isRateType_; }

    unsigned int getRank() { return rank_; }
protected:
    unsigned int nbNeurons_; 	///< amount of neurons in the layer
    std::string name_;  		///< name of layer
    int maxDelay_;				///< maximum delay over all neurons. Please note, that this value is set after analyzing the attached projections.
    DATA_TYPE dt_;				///< discretization constant
    bool isRateType_;			///< is the current population rate coded.
    std::vector< std::vector<class Dendrite*> > dendrites_; ///< list of afferent dendrites ordered neuron wise
    std::vector< std::vector< std::vector<class Dendrite*> > > typedDendrites_;	///< list of afferent dendrites ordered neuron and type wise, to improve performance of the weighted sum.

    unsigned int rank_ = 0;				///< internal identifier of the population
};

/*TODEL
 *
 * Old implementation of population class containing both mean and spike coding.
 *
class Population{
public:
	// functions
	Population(std::string name, int nbNeurons);

	virtual ~Population();

	virtual void prepareNeurons() {};
	virtual void metaSum();
	virtual void localMetaStep(int neur_rank) {};
	virtual void globalMetaStep() {};
	virtual void globalOperations();
	virtual void record() {}
	virtual void resetToInit() {}

	void metaLearn();

	void metaStep();

	std::string getName() { return name_; }

	virtual int getNeuronCount() { return nbNeurons_; }

	std::vector<class Projection*> getProjections(int neuron, int type);

	class Projection* getProjection(int neuron, int type, Population* pre);

    void addSpikeTarget(Projection* proj);

	void addProjection(int postRankID, Projection* proj);

	void removeProjection(Population *pre);

	void printRates();

	void setMaxDelay(int delay);

	DATA_TYPE sum(int neur, int type);

	std::vector<DATA_TYPE>* getRates() {
		return &rate_;
	}

	std::vector<DATA_TYPE>* getRates(int delay)
	{
		if ( delay <= (int)delayedRates_.size()) 
        {
        #ifdef _DEBUG
            std::cout << name_ << ": rates for delay "<< delay << "(" << maxDelay_ <<")" << std::endl;
            for(int i=0; i < delayedRates_.size(); i++)
                    std::cout << "   data-addr: " << &(delayedRates_[i]) << std::endl;
        #endif
            return &(delayedRates_[delay-1]);
        }
		else
        {
            std::cout << "Invalid delay " << delay << " (maxDelay is "<< maxDelay_ << ")"<< std::endl;
            return NULL;
        }
    }

	std::vector< std::vector<int> > getSpikeTimings() { return spike_timings_;}

	std::vector<DATA_TYPE> getRates(std::vector<int> delays, std::vector<int> ranks);

	DATA_TYPE getDt() { return dt_;	}

	void setDt(DATA_TYPE dt) { dt_ = dt; }

	int getLastSpikeTime(int rank)
	{
	    if(spike_timings_[rank].empty())
	        return 0;
	    else
	        return spike_timings_[rank].back();
	}
#ifdef ANNAR_PROFILE
    FILE *cs;
    FILE *gl;
    FILE *ll;
#endif


protected:
#ifdef ANNAR_SCHEDULE
    // coreCounter counts the runtime of each thread on each CPU
    // and the number of switches of a thread between the CPUs
    int * volatile *coreCounter;
#endif

	// data
	int nbNeurons_;
	std::string name_;	///< name of layer
	int maxDelay_;
	DATA_TYPE dt_;

	std::vector<DATA_TYPE>	rate_;
	std::deque< std::vector<DATA_TYPE> > delayedRates_;
	std::vector< std::vector<class Projection*> > projections_;	// first dimension, neuron wise
	std::vector< std::vector<class Projection*> > spikeTargets_; // first dimension, neuron wise

	std::vector< bool > spiked_;
	std::vector< std::vector<int> > spike_timings_;
};
*/
#endif
