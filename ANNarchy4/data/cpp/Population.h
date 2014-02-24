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
#ifndef __POPULATION_H__
#define __POPULATION_H__

#include "Global.h"

class Population
{
public:
    Population( std::string name, int nbNeurons, bool isRateType );

    virtual ~Population();

    std::string getName() { return name_; }

    int getNeuronCount() { return nbNeurons_; }

    DATA_TYPE getDt() { return dt_; }

    void setDt(DATA_TYPE dt) { dt_ = dt; }

    //
    //  Projection handling
    //
    void addProjection(int postRankID, class Projection* proj);

    void removeProjection(Population *pre);

    class Projection* getProjection(int neuron, int type, Population* pre);

    std::vector<class Projection*> getProjections(int neuron, int type);

    //
    //  pure virtuals
    //
    bool isMeanRateCoded() { return isRateType_; }
protected:
    int nbNeurons_; ///< amount of neurons in the layer
    std::string name_;  ///< name of layer
    int maxDelay_;
    DATA_TYPE dt_;
    bool isRateType_;

    std::vector< std::vector<class Projection*> > projections_; ///< list of afferent dendrites ordered neuron wise
};

/*
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
