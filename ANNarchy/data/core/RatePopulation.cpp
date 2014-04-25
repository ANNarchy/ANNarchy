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
#include "Global.h"
#include "RatePopulation.h"
#include "RateDendrite.h"

RatePopulation::RatePopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, true)
{
    rate_ = std::vector<DATA_TYPE>(nbNeurons_, 0.0);
    delayedRates_ = std::deque< std::vector<DATA_TYPE> >();
}

RatePopulation::~RatePopulation()
{

}

std::vector<DATA_TYPE>* RatePopulation::getRates()
{
    return &rate_;
}

std::vector<DATA_TYPE>* RatePopulation::getRates(int delay)
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

std::vector<DATA_TYPE> RatePopulation::getRates(std::vector<int> delays, std::vector<int> ranks)
{
    std::vector<DATA_TYPE> vec = std::vector<DATA_TYPE>(delays.size(), 0.0);

    if(delays.size() != ranks.size()) {
        std::cout << "Invalid vector ranges. " << std::endl;
        return std::vector<DATA_TYPE>();
    }

    for(unsigned int n = 0; n < ranks.size(); n++) {
        vec[n] = delayedRates_[ranks[n]][delays[n]-1];
    }

    return vec;
}

void RatePopulation::setMaxDelay(int delay)
{
    // TODO:
    // maybe we should take the current fire rate as initial value
#ifdef _DEBUG
    std::cout << "Population " << name_ << " got new max delay: " << delay << std::endl;
#endif
    if(delay > maxDelay_)
    {
        for(int oldSize = delayedRates_.size(); oldSize < delay; oldSize++)
            delayedRates_.push_front(std::vector<DATA_TYPE>(nbNeurons_, (DATA_TYPE)oldSize));

        maxDelay_ = delay;
    }

#ifdef _DEBUG
    std::cout << "current delay vec: " << delayedRates_.size() << std::endl;
    for(int i=0; i<delayedRates_.size(); i++)
            std::cout << "   Delay: " << i << " rates: " << delayedRates_[i].size() << std::endl;
#endif
}

DATA_TYPE RatePopulation::sum(int neur, int typeID)
{
    DATA_TYPE sum=0.0;

#ifdef _DEBUG
    if ( neur >= typedDendrites_.size() )
    {
    	std::cout << "No dendrite with id = " << neur << std::endl;
    	return sum;
    }

    if ( typeID >= typedDendrites_[neur].size() )
    {
    	std::cout << "No target with id = " << typeID << std::endl;
    	return sum;
    }
#endif
    auto it = typedDendrites_[neur][typeID].begin();
    int end = typedDendrites_[neur][typeID].size();

    for(int i=0; i != end; i++ )
        sum += static_cast<class RateDendrite*>(*(it++))->getSum();

    return sum;
}

void RatePopulation::metaSum()
{
#ifdef ANNAR_PROFILE
    double start = 0, stop = 0;
    #pragma omp barrier

    #pragma omp master
    {
        start = omp_get_wtime();
    }
#endif

#ifdef _DEBUG
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Meta sum                #"<< std::endl;
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
        for(unsigned int p = 0; p < dendrites_[n].size();p++)
        {
        #ifdef _DEBUG
            std::cout << "reference: " << dendrites_[n][p] << std::endl;
            std::cout << "\tpost = " << name_ << std::endl;
            std::cout << "\tpre = " << dendrites_[n][p]->getPrePopulation()->getName() << std::endl;
            std::cout << "\tsynaseCount = " << (int)(dendrites_[n][p]->getSynapseCount()) << std::endl;
        #endif
           	(static_cast<class RateDendrite*>(dendrites_[n][p]))->computeSum();
        }

    #ifdef ANNAR_SCHEDULE
        // increase the number of runs for the current thread on the current scheduled cpu
        coreCounter[omp_get_thread_num()][sched_getcpu()]++;
        // if the last scheduled cpu is different from the actual scheduled cpu then increase
        // the number of switches for the current thread
        if (coreCounter[omp_get_thread_num()][omp_get_num_procs() + 1] != sched_getcpu()) {
            coreCounter[omp_get_thread_num()][omp_get_num_procs() + 1] = sched_getcpu();
            coreCounter[omp_get_thread_num()][omp_get_num_procs()]++;
        }
    #endif

    }

    #pragma omp barrier
#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        stop = omp_get_wtime();

        Profile::profileInstance()->appendTimeSum(name_, (stop-start)*1000.0);
    }
#endif
}

void RatePopulation::metaStep()
{
    double start, stop = 0.0;

#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        start = omp_get_wtime();
    }
#endif

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

#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        stop = omp_get_wtime();
        Profile::profileInstance()->appendTimeStep(name_, (stop-start)*1000.0);
    }
#endif

#ifdef _DEBUG
    #pragma omp master
    {
        std::cout << "local computation done."<< std::endl;
    }
#endif

}

void RatePopulation::metaLearn()
{
    double start = 0.0, stop = 0.0;

#ifdef ANNAR_PROFILE
    #pragma omp barrier

    start = omp_get_wtime();
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

        for(unsigned int p = 0; p < dendrites_[n].size();p++)
        {
            if ( dendrites_[n][p]->isLearning() )
            	static_cast<class RateDendrite*>(dendrites_[n][p])->globalLearn();
        }
    }

    #pragma omp barrier

#ifdef ANNAR_PROFILE
    stop = omp_get_wtime();
	#pragma omp master
	{
        Profile::profileInstance()->appendTimeGlobal(name_, (stop-start)*1000.0);
    }
	#pragma omp barrier
	start = omp_get_wtime();
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
            	static_cast<class RateDendrite*>(dendrites_[n][p])->localLearn();
        }
    }

    #pragma omp barrier
#ifdef ANNAR_PROFILE
    stop = omp_get_wtime();
    #pragma omp master
    {
        Profile::profileInstance()->appendTimeLocal(name_, (stop - start)*1000.0);
    }
#endif
}
