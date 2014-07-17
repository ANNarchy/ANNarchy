/*
 *    SpikeProjection.h
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
#ifndef __ANNARCHY_SPIKE_PROJECTION_H__
#define __ANNARCHY_SPIKE_PROJECTION_H__

#include "Global.h"

class SpikeProjection : public Projection
{
public:
	SpikeProjection();

    void globalLearn();

    void localLearn();

    void postEvent(std::vector<int> post_ranks);

    void evaluatePostEvents();

    void evaluatePreEvents();

    void computePsp();

	Population* getPrePopulation();

	class Dendrite *getDendrite(int postNeuronRank);

	void removeDendrite(int postNeuronRank, class Population *pre);

	bool isRateCoded() { return false; }

	void record();

	int nbDendrites() { return dendrites_.size();}

	int nbSynapses(int post_rank);

protected:
	int nbDendrites_;
	std::vector<int> post_spikes_;
	std::deque<std::vector<int> > delayed_post_spikes_;

	class SpikePopulation* pre_population_;
	class SpikePopulation* post_population_;
};
#endif
