/*
 *    Network.h
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
#ifndef __ANNARCHY_NETWORK_H__
#define __ANNARCHY_NETWORK_H__

#include "Global.h"

class Network {
public:
	// functions
	static Network* instance() {
		if(instance_==NULL) {
			instance_ = new Network();
		}

		return instance_;
	}

	~Network();

	class Population* getPopulation(std::string name);

	class Population* getPopulation(std::string name, bool isRateCoded);

	class Population* getPopulation(unsigned int id);

	class Population* getPopulation(unsigned int id, bool isRateCoded);

	void setNumThreads(int threads)
	{
	    omp_set_num_threads(threads);
	}

	void connect(int prePopulationID, int postPopulationID, int projectionID, int target, bool spike, std::string filename);

	void disconnect(int prePopulationID, int postPopulationID, bool preIsSpike, bool postIsSpike, int target=-1);

	void run(int steps);

	/**
	 * 	\brief		add a population to the network
	 * 	\details	will be automatically called by the Population::Population() method
	 */
	void addPopulation(class Population* population);

	int getTime() { return ANNarchy_Global::time; }

	void setTime(int time) { ANNarchy_Global::time = time; }
protected:
	Network();

private:
	static Network *instance_;

	// data
	std::vector<class RatePopulation*> rate_populations_;
	std::vector<class SpikePopulation*> spike_populations_;
};
#endif
