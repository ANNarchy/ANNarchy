/*
 *    SpikeDendrite.h
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
#ifndef __ANNARCHY_SPIKE_DENDRITE_H__
#define __ANNARCHY_SPIKE_DENDRITE_H__

#include "Global.h"

class SpikeDendrite: public Dendrite
{
public:
	SpikeDendrite(class SpikeProjection* proj);

    /**
     * 	\brief		returns the reference to presynaptic populations
     * 	\details	is abstract, cause the reference is hold by child class.
     */
	virtual class Population* getPrePopulation() = 0;

	virtual int addSynapse(int rank, DATA_TYPE w, int delay) = 0;

    virtual int removeSynapse(int rank) = 0;

    virtual int removeAllSynapses() = 0;

    virtual void initValues() = 0;

    virtual void record() = 0;

	virtual void globalLearn() = 0;

	virtual void localLearn() = 0;

    virtual void preEventPsp(int rank) = 0;

    virtual void preEventLearn(int rank) = 0;

    virtual void postEvent() = 0;

protected:
};

#endif
