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
#ifndef __NETWORK_H__
#define __NETWORK_H__

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

	class Population* getPopulation(unsigned int id);

	void setNumThreads(int threads)
	{
	    omp_set_num_threads(threads);
	}

	/*
	std::vector<DATA_TYPE> getRates(int populationID);

	std::vector<DATA_TYPE> getRates(int populationID,int delay);

	std::vector<DATA_TYPE> getRates(int populationID, std::vector<int> delays, std::vector<int> ranks);
    */

	void connect(int prePopulationID, int postPopulationID, class Connector *connector, int projectionID, int target);

	void disconnect(int prePopulationID, int postPopulationID, int target=-1);

	void run(int steps);

	void addPopulation(class Population* population);

	int getTime() { return ANNarchy_Global::time; }

	void setTime(int time) { ANNarchy_Global::time = time; }
protected:
	Network();

private:
	static Network *instance_;

	// data
	std::vector<class Population*> populations_;

	std::vector<class MeanPopulation*> mean_populations_;
	std::vector<class SpikePopulation*> spike_populations_;
};
#endif
