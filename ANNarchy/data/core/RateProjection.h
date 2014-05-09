/*
 *    RateProjection.h
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
#ifndef __ANNARCHY_RATE_PROJECTION_H__
#define __ANNARCHY_RATE_PROJECTION_H__

#include "Global.h"

class RateProjection : public Projection
{
public:
	RateProjection(std::string pre, std::string post, int target);

	RateProjection(Population *pre, Population* post, int target);

	void computeSum();

    void globalLearn();

    void localLearn();

    DATA_TYPE getSum(int neuron);

	virtual Population* getPrePopulation() { return NULL; }

	void addDendrite(int postNeuronRank, class Dendrite *dendrite);

	virtual void addDendrite(int postNeuronRank, std::vector<int> ranks, std::vector<DATA_TYPE> values, std::vector<int> delays) {}

	class Dendrite *getDendrite(int postNeuronRank);

	void removeDendrite(int postNeuronRank, class Population *pre);

	bool isRateCoded() { return true; }

	void record() {}

protected:
	int nbDendrites_;

	std::vector< class RateDendrite* > dendrites_;
};
#endif
