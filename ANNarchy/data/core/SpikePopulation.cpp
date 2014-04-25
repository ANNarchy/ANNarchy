/*
 *    SpikePopulation.cpp
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
#include "SpikePopulation.h"
#include "SpikeDendrite.h"

SpikePopulation::SpikePopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, false)
{
	spiked_ = std::vector< bool >(nbNeurons_, false);
	spike_timings_ = std::vector< std::vector<int> >(nbNeurons_, std::vector<int>() );

	spikeTargets_ = std::vector<std::vector<Dendrite*> >(nbNeurons_, std::vector<Dendrite*>());
}

SpikePopulation::~SpikePopulation()
{

}

void SpikePopulation::addSpikeTarget(Dendrite* dendrite)
{
#ifdef _DEBUG
    std::cout << name_ << ": added dendrite as spike target " << std::endl;
    std::cout << "address " << dendrite << std::endl;
#endif
    for(unsigned int n=0; n< nbNeurons_; n++)
    {
        spikeTargets_.at(n).push_back(dendrite);
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
        if ( dendrites_[n].size() > 0 && omp_get_thread_num() == 0 )
            std::cout << name_<<"("<< n << "): "<< dendrites_[n].size()<< " projections."<< std::endl;
    #endif
        for(unsigned int p=0; p < dendrites_[n].size();p++)
        {
        	if ( dendrites_[n][p]->isLearning() )
        		static_cast<class SpikeDendrite*>(dendrites_[n][p])->globalLearn();
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
        if ( dendrites_[n].size() > 0 && omp_get_thread_num() == 0 )
            std::cout << name_<<"("<< n << "): "<< dendrites_[n].size()<< " projections."<< std::endl;
    #endif
        for(unsigned int p = 0; p < dendrites_[n].size();p++) {
        	if ( dendrites_[n][p]->isLearning() )
        		static_cast<class SpikeDendrite*>(dendrites_[n][p])->localLearn();
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
