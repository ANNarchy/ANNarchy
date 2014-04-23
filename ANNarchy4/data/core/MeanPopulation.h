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
#ifndef __MEAN_POPULATION_H__
#define __MEAN_POPULATION_H__

#include "Global.h"

/**
 *  \brief      Implementation of mean rate coded populations.
 *  \details    base class for all mean rate implementations, inherited from
 *              Population and will be inherited from Population0..PopulationN,
 *              the population classes generated from the users input.
 */
class MeanPopulation: public Population
{
public:
    /**
     *  \brief      Constructor
     *  \details    Initializes the mean rate arrays and calls the Population
     *              constructor.
     */
    MeanPopulation(std::string name, int nbNeurons);

    /**
     *  \brief      Destructor
     *  \details    Destroys the attached data.
     */
    virtual ~MeanPopulation();

    /**
     *  \brief      set max delay
     */
    void setMaxDelay(int delay);

    DATA_TYPE sum(int neur, int type);

    std::vector<DATA_TYPE>* getRates();

    std::vector<DATA_TYPE>* getRates(int delay);

    std::vector<DATA_TYPE> getRates(std::vector<int> delays, std::vector<int> ranks);

    /**
     *  \brief      evaluation of summing up presynaptic inputs, called by Network::run
     */
    void metaSum();

    /**
     *  \brief      evaluation of neuron equations, called by Network::run
     */
    void metaStep();

    /**
     *  \brief      evaluation of learning rule, called by Network::run
     */
    void metaLearn();

    // override
    virtual void localMetaStep(int neur_rank) {};
    virtual void globalMetaStep() {};
    virtual void globalOperations() {};

    virtual void record() {}
    virtual void resetToInit() {}

protected:
    std::vector<DATA_TYPE>  rate_;
    std::deque< std::vector<DATA_TYPE> > delayedRates_;
};


#endif
