#ifndef __MEAN_POPULATION_H__
#define __MEAN_POPULATION_H__

#include "Global.h"

class MeanPopulation: public Population
{
public:
    MeanPopulation(std::string name, int nbNeurons);

    virtual ~MeanPopulation();

    // override
    virtual void localMetaStep(int neur_rank) {};
    virtual void globalMetaStep() {};
    virtual void globalOperations() {};

    virtual void record() {}
    virtual void resetToInit() {}

    // access methods
    void setMaxDelay(int delay);

    DATA_TYPE sum(int neur, int type);

    std::vector<DATA_TYPE>* getRates();

    std::vector<DATA_TYPE>* getRates(int delay);

    std::vector<DATA_TYPE> getRates(std::vector<int> delays, std::vector<int> ranks);

    // evaluation of learning rule, called by Network::run
    void metaSum();
    void metaStep();
    void metaLearn();

protected:
    std::vector<DATA_TYPE>  rate_;
    std::deque< std::vector<DATA_TYPE> > delayedRates_;
};


#endif
