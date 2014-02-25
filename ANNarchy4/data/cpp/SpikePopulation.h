#ifndef __SPIKE_POPULATION_H__
#define __SPIKE_POPULATION_H__

#include "Global.h"

class SpikePopulation: public Population
{
public:
    SpikePopulation(std::string name, int nbNeurons);

    virtual ~SpikePopulation();

    void addSpikeTarget(class Projection* proj);

	std::vector< std::vector<int> > getSpikeTimings() { return spike_timings_;}

	int getLastSpikeTime(int rank);

	void setMaxDelay(int delay);

	void metaStep();
	void metaLearn();

	// override
	virtual void prepareNeurons() {}
	virtual void record() {}
	virtual void globalMetaStep() {}
	virtual void localMetaStep(int rank) {}
	virtual void globalOperations() {}

protected:
	std::vector< std::vector<class Projection*> > spikeTargets_; // first dimension, neuron wise

	std::vector< bool > spiked_;
	std::vector< std::vector<int> > spike_timings_;
};


#endif
