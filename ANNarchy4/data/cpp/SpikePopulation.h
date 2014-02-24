#ifndef __SPIKE_POPULATION_H__
#define __SPIKE_POPULATION_H__

#include "Global.h"

class SpikePopulation: public Population
{
public:
    SpikePopulation(std::string name, int nbNeurons);

    virtual ~SpikePopulation();

    // override

protected:

};


#endif
