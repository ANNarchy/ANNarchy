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
