#ifndef __SPIKE_POPULATION_H__
#define __SPIKE_POPULATION_H__

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
    void addSpikeTarget(class Projection* projection);

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
	std::vector< std::vector<class Projection*> > spikeTargets_; // first dimension, neuron wise

	std::vector< bool > spiked_;
	std::vector< std::vector<int> > spike_timings_;
};


#endif
