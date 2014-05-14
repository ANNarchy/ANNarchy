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
#ifndef __ANNARCHY_PROJECTION_H__
#define __ANNARCHY_PROJECTION_H__

#include "Global.h"

class Projection
{
public:
	Projection();

    ~Projection();

    virtual void globalLearn() = 0;

    virtual void localLearn() = 0;

    virtual Population* getPrePopulation() = 0;

    virtual void addDendrite(int postNeuronRank, class Dendrite *dendrite) = 0;

    virtual void addDendrite(int postNeuronRank, std::vector<int> ranks, std::vector<DATA_TYPE> values, std::vector<int> delays) = 0;

    virtual class Dendrite* getDendrite(int postNeuronRank) = 0;

    virtual void removeDendrite(int postNeuronRank, class Population *pre) = 0;

    virtual bool isRateCoded() = 0;

    virtual void record() = 0;

    virtual void initValues(int postNeuronRank) = 0;

    bool isLearning() { return isLearning_; }

    int getTarget() { return target_; }

protected:
    bool isLearning_;
    int target_;
};
#endif
