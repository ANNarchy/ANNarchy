#include "SpikePopulation.h"

SpikePopulation::SpikePopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, false)
{
	spiked_ = std::vector< bool >(nbNeurons_, false);
	spike_timings_ = std::vector< std::vector<int> >(nbNeurons_, std::vector<int>() );

	spikeTargets_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());
}

SpikePopulation::~SpikePopulation()
{

}

void SpikePopulation::addSpikeTarget(Projection* proj)
{
#ifdef _DEBUG
    std::cout << name_ << ": added projection as spike target " << std::endl;
#endif
    for(unsigned int n=0; n< nbNeurons_; n++)
    {
        spikeTargets_.at(n).push_back(proj);
    }
}

int SpikePopulation::getLastSpikeTime(int rank)
{
    if(spike_timings_[rank].empty())
        return 0;
    else
        return spike_timings_[rank].back();
}

void SpikePopulation::setMaxDelay(int delay)
{
	//
	// TODO:
}

void SpikePopulation::metaStep()
{
    // Random generators
    #pragma omp master
    {
        globalMetaStep();
    } // end of master region
    #pragma omp barrier

#ifdef _DEBUG
    #pragma omp master
    {
        std::cout << "global computation done."<< std::endl;
    }
#endif

    #pragma omp for
    for(int i=0; i<nbNeurons_; i++)
    {
        localMetaStep(i);
    }
    #pragma omp barrier

#ifdef _DEBUG
    #pragma omp master
    {
        std::cout << "local computation done."<< std::endl;
    }
#endif
}
