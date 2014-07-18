/*
 *    RatePopulation.cpp
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
#include "RateProjection.h"

RatePopulation::RatePopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, true)
{
    r_ = std::vector<DATA_TYPE>(nbNeurons_, 0.0);

#ifdef _DEBUG
    std::cout << "Rate reference: " << &r_ << std::endl;
#endif
    delayedRates_ = std::deque< std::vector<DATA_TYPE> >();
}

RatePopulation::~RatePopulation()
{

}

std::vector<DATA_TYPE>* RatePopulation::getRs()
{
    return &r_;
}

std::vector<DATA_TYPE>* RatePopulation::getRs(int delay)
{
    if ( delay <= (int)delayedRates_.size())
    {
    #ifdef _DEBUG_DELAY
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

std::vector<DATA_TYPE> RatePopulation::getRs(std::vector<int> delays, std::vector<int> ranks)
{
	std::vector<DATA_TYPE> vec = std::vector<DATA_TYPE>(delays.size(), 0.0);

    if(delays.size() != ranks.size()) {
        std::cout << "Invalid vector ranges. " << std::endl;
        return std::vector<DATA_TYPE>();
    }

    for(unsigned int n = 0; n < ranks.size(); n++) {
        vec[n] = delayedRates_[delays[n]-1][ranks[n]];
    }

    return vec;
}

void RatePopulation::setMaxDelay(int delay)
{
#ifdef _DEBUG_DELAY
    std::cout << "Population " << name_ << " got new max delay: " << delay << std::endl;
#endif
    if( delay > maxDelay_ )
    {
        for(int oldSize = delayedRates_.size(); oldSize < delay; oldSize++)
            delayedRates_.push_front( r_ );

        maxDelay_ = delay;
    }

#ifdef _DEBUG_DELAY
    std::cout << "current delay vec: " << delayedRates_.size() << std::endl;
    for(int i=0; i<delayedRates_.size(); i++)
            std::cout << "   Delay: " << i << " rates: " << delayedRates_[i].size() << std::endl;
#endif
}

DATA_TYPE RatePopulation::sum(int neur, int typeID)
{
    DATA_TYPE sum=0.0;

#ifdef _DEBUG
    if ( typeID >= typedProjections_.size() )
    {
    	std::cout << "No projection with target id = " << typeID << std::endl;
    	return sum;
    }
#endif
    auto it = typedProjections_[typeID].begin();
    int end = typedProjections_[typeID].size();

    for(int i=0; i < end; i++ )
        sum += static_cast<class RateProjection*>(*(it++))->getSum(neur);

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

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Meta sum                #"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
#endif
#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
	if ( projections_.size() > 0 && omp_get_thread_num() == 0 )
		std::cout << name_ <<": "<< projections_.size()<< " projection(s)"<< std::endl;
#endif

	//
	// parallelization will be inside
	for(unsigned int p = 0; p < projections_.size();p++)
	{
    	static_cast<RateProjection*>(projections_[p])->computeSum();
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
#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
#pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Meta step               #"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
#endif
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

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
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

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
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

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Global learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
    #pragma barrier
#endif

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
	if ( projections_.size() > 0 && omp_get_thread_num() == 0 )
		std::cout << name_<<": "<< projections_.size()<< " projections."<< std::endl;
#endif
	for(unsigned int p = 0; p < projections_.size(); p++)
	{
		if ( projections_[p]->isLearning() )
		{
			static_cast<class RateProjection*>(projections_[p])->globalLearn();
		}
		#pragma omp barrier
	}

#ifdef ANNAR_PROFILE
    stop = omp_get_wtime();
	#pragma omp master
	{
        Profile::profileInstance()->appendTimeGlobal(name_, (stop-start)*1000.0);
    }
	#pragma omp barrier
	start = omp_get_wtime();
#endif

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Local  learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
#endif

#if defined( _DEBUG ) && defined ( _DEBUG_SIMULATION_CONTROL )
	if ( projections_.size() > 0 && omp_get_thread_num() == 0 )
		std::cout << name_<<": "<< projections_.size()<< " projections."<< std::endl;
#endif
	for(unsigned int p = 0; p < projections_.size(); p++)
	{
		if ( projections_[p]->isLearning() )
		{
			static_cast<class RateProjection*>(projections_[p])->localLearn();
		}
		#pragma omp barrier
	}

#ifdef ANNAR_PROFILE
    stop = omp_get_wtime();
    #pragma omp master
    {
        Profile::profileInstance()->appendTimeLocal(name_, (stop - start)*1000.0);
    }
#endif
}
