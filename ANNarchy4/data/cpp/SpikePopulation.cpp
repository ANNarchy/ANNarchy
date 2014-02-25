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

void SpikePopulation::metaLearn()
{
    double start = 0.0, stop = 0.0;

#ifdef ANNAR_PROFILE
    #pragma omp barrier

    #pragma omp master
    {
        double start = omp_get_wtime();
    }
#endif

#ifdef _DEBUG
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Global learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
    #pragma barrier
#endif
    #pragma omp for
    for(int n=0; n<nbNeurons_; n++)
    {
    #ifdef _DEBUG
        if ( projections_[n].size() > 0 && omp_get_thread_num() == 0 )
            std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++)
        {
            projections_[n][p]->globalLearn();
        }
    }

    #pragma omp barrier

#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        stop = omp_get_wtime();

        Profile::profileInstance()->appendTimeGlobal(name_, (stop-start)*1000.0);
        start = omp_get_wtime();
    }
#endif

#ifdef _DEBUG
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Local  learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
#endif

    #pragma omp for
    for(int n=0; n<nbNeurons_; n++)
    {
    #ifdef _DEBUG
        if ( projections_[n].size() > 0 && omp_get_thread_num() == 0 )
            std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->localLearn();
        }
    }

    #pragma omp barrier
#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        stop = omp_get_wtime();

        Profile::profileInstance()->appendTimeLocal(name_, (stop - start)*1000.0);
    }
#endif
}
