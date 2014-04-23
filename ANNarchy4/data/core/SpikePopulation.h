/*
 *    SpikePopulation.h
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
#ifndef __ANNARCHY_SPIKE_POPULATION_H__
#define __ANNARCHY_SPIKE_POPULATION_H__

#include "Global.h"

/**
 *  \brief      Implementation of spike coded populations
 *  \details    base class for all spike implementations, inherited from
 *              Population and will be inherited from Population0..PopulationN,
 *              the population classes generated from the users input.
 */
class SpikePopulation: public Population
{
public:
    /**
     *  \brief      Constructor
     *  \details    Initializes the spike arrays and calls the Population constructor.
     */
    SpikePopulation(std::string name, int nbNeurons);

    /**
     *  \brief      Destructor
     *  \details    Destroys the attached data.
     */
    virtual ~SpikePopulation();

    /**
     * \brief       Add a spike target.
     * \details     Neurons of this population will emit a spike to the provided
     *              projection in case of spike event.
     * \param[IN]   projection  Instance of Projection determing the spike target.
     *              A spike will activate the Projection::preEvent method.
     */
    void addSpikeTarget(class Dendrite* dendrite);

    /**
     * \brief       Get the spike times of all neurons.
     * \details     Sometimes it may useful to trace all the emited spikes, e.g. visualization.
     *              Please note:
     *              #   that the spike times will reseted through call of
     *                  Population::resetToInit method
     *              #   that the spike times are an offset to the last reset time
     *                  point
     * \param[OUT]  1st dimension: neurons sorted by their rank
     *              2nd dimension: spike times of the neuron
     */
	std::vector< std::vector<int> > getSpikeTimings() { return spike_timings_;}

    /**
     * \brief       Get the laste spike time of a neuron.
     * \details     Sometimes it may useful to trace all the emited spikes, e.g.
     *              visualization.
     * \param[OUT]  last spike time of neuron.
     */
	int getLastSpikeTime(int rank);

    /**
     *  \brief      set max delay
     *  \details    NOT IMPLEMENTED
     */
	void setMaxDelay(int delay);

    /**
     *  \brief      evaluation of neuron equations, called by Network::run
     */
	void metaStep();

    /**
     *  \brief      evaluation of learning rule, called by Network::run
     */
	void metaLearn();

	// override
	virtual void prepareNeurons() {}
	virtual void resetToInit() {}
	virtual void record() {}
	virtual void globalMetaStep() {}
	virtual void localMetaStep(int rank) {}
	virtual void globalOperations() {}

protected:
	std::vector< std::vector<class Dendrite*> > spikeTargets_; // first dimension, neuron wise

	std::vector< bool > spiked_;
	std::vector< std::vector<int> > spike_timings_;
};


#endif
