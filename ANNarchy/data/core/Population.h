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
    void addProjection(class Projection* projection);

    /**
     *  \brief		remove a projection to another presynaptic population
     *  \param[in]	pre		presynaptic population
     *  \param[in]	target 	integer target ID
     */
    void removeProjections(class Population* pre);

    /**
     *  \brief		remove a projection to another presynaptic population
     *  \param[in]	pre		presynaptic population
     *  \param[in]	target 	integer target ID
     */
    void removeProjection(class Population* pre, int target);

    /**
     *  \brief		get a projection to another presynaptic population
     *  \param[in]	pre		presynaptic population
     *  \param[in]	target 	integer target ID
     *  \return		reference to the projection, if one exists with these parameters, otherwise NULL.
     */
    class Projection* getProjection(class Population* pre, int target);


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
    std::vector< class Projection*> projections_;
    std::vector< std::vector< class Projection*> > typedProjections_;

    unsigned int rank_;				///< internal identifier of the population
};

#endif
