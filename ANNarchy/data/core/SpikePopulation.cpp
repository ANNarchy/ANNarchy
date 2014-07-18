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
#include "SpikeProjection.h"

SpikePopulation::SpikePopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, false)
{
	spike_timings_ = std::vector< std::vector<int> >(nbNeurons_, std::vector<int>() );

	spikeTargets_ = std::vector<std::vector<Dendrite*> >(nbNeurons_, std::vector<Dendrite*>());

    record_spike_ = false;

    lastSpike_ = std::vector<int>(nbNeurons_, -10000);
    spiked = std::vector<bool>(nbNeurons_, false);

    has_refractory_ = false;
    refractory_counter_ = std::vector<int>(nbNeurons_,  0);
    refractory_times_   = std::vector<int>(nbNeurons_,  0);
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
    return lastSpike_[rank];
}

bool SpikePopulation::hasSpiked(int rank, int t) 
{ 
    if (t==-1){ // asking for the current step
        return spiked[rank];
    }
    else{
        for(int i=spike_timings_[rank].size()-1; i>0; i--){
            if(spike_timings_[rank][i] == t)
                return true;
            if(spike_timings_[rank][i] < t)
                return false;
        }
        return false;
    }
}

void SpikePopulation::emit_spike(int i)
{

    if (refractory_counter_[i] < 1)
    {
        #pragma omp critical
        {
            //std::cout << "emit spike (pop " << name_ <<")["<<i<<"] ( time="<< ANNarchy_Global::time<< ")" << std::endl;
            this->propagate_.push_back(i);
            
            spiked[i] = true;
            lastSpike_[i] = ANNarchy_Global::time;
            if(record_spike_){
                spike_timings_[i].push_back(ANNarchy_Global::time);
            }
        }
    }
}

int SpikePopulation::nbSpikesInTheLast(int rank, int t) 
{ 
    if(!record_spike_)
        return 0;
    int nb = 0;
    for(int i=spike_timings_[rank].size()-1; i>0; i--){
        if(spike_timings_[rank][i] < ANNarchy_Global::time - t)
            return nb;
        nb++;
    }
    return nb;

}


void SpikePopulation::reset_spike_timings()
{
    int last_spike;
    for(int i=0; i<spike_timings_.size(); i++)
    {
        if (!spike_timings_[i].empty())
        {
            spike_timings_[i].clear();
        }
    }
}

void SpikePopulation::start_record_spike()
{
    record_spike_=true;
}
void SpikePopulation::stop_record_spike()
{
    record_spike_=false;
}
void SpikePopulation::setMaxDelay(int delay)
{
	//
	// TODO:
}

void SpikePopulation::metaStep()
{
	propagate_.clear();
    std::fill(spiked.begin(), spiked.end(), false);

    // Random generators
    #pragma omp master
    {
        globalMetaStep();
    } // end of master region
    #pragma omp barrier

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
    #pragma omp master
    {
        std::cout << "global computation done."<< std::endl;
    }
#endif

    #pragma omp for
    for(int n=0; n<nbNeurons_; n++)
    {
    	localMetaStep(n);

    	if (refractory_counter_[n] > 0)
    		reset(n);
    }
    #pragma omp barrier

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
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

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Global learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
    #pragma barrier
#endif

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
	if ( projections_.size() > 0 && omp_get_thread_num() == 0 )
		std::cout << name_<<": "<< projections_.size()<< " projections."<< std::endl;
#endif
	for(unsigned int p=0; p < projections_.size();p++)
	{
		if ( projections_[p]->isLearning() )
			static_cast<class SpikeProjection*>(projections_[p])->globalLearn();
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

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
    #pragma omp master
    {
    std::cout << "###########################"<< std::endl;
    std::cout << "# Local  learning         #"<< std::endl;
    std::cout << "# Population '"<< name_ <<"'#"<< std::endl;
    std::cout << "###########################"<< std::endl;
    }
#endif

#if defined(_DEBUG) && defined(_DEBUG_SIMULATION_CONTROL)
	if ( projections_.size() > 0 && omp_get_thread_num() == 0 )
		std::cout << name_<<": "<< projections_.size()<< " projections."<< std::endl;
#endif
    for(unsigned int p=0; p < projections_.size();p++)
	{
		if ( projections_[p]->isLearning() )
			static_cast<class SpikeProjection*>(projections_[p])->localLearn();
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

void SpikePopulation::updateRefractoryCounter()
{
#ifdef _DEBUG
	#pragma omp master
	{
		std::cout << "updateRefactoryCounter:"<< std::endl;
		for( auto it = refractory_counter_.begin(); it != refractory_counter_.end(); it++ )
			std::cout << *it << " ";
		std::cout << std::endl;
	}
#endif
	#pragma omp master
	{
		for( int n = 0; n < nbNeurons_; n++ )
		{
			refractory_counter_[n] -= 1;
		}
	}
}
void SpikePopulation::propagateSpikes() 
{
#ifdef ANNAR_PROFILE
    double start, stop;
    start = omp_get_wtime();
#endif
    if (!propagate_.empty())
    {
        // emit a postsynaptic spike on receiving projections
        for( auto p_it = projections_.begin(); p_it != projections_.end(); p_it++)
        {
            if ( static_cast<SpikeProjection*>(*p_it)->isLearning() )
                static_cast<SpikeProjection*>(*p_it)->postEvent(propagate_);
        }

        /*
        for(auto n_it= propagate_.begin(); n_it!= propagate_.end(); n_it++)
        {
            // emit a presynaptic spike on outgoing projections
            for( auto p_it = spikeTargets_[(*n_it)].begin(); p_it != spikeTargets_[(*n_it)].end(); p_it++)
            {
                static_cast<SpikeDendrite*>(*p_it)->preEvent(*n_it);
            }
        }
    
        // spike handling is completed
        propagate_.erase(propagate_.begin(), propagate_.end());
        */
    }
#ifdef ANNAR_PROFILE
    stop = omp_get_wtime();
    Profile::profileInstance()->appendTimeSpikeDelivery(name_, (stop - start)*1000.0);
#endif
}

void SpikePopulation::evaluatePostSpikes()
{
#ifdef ANNAR_PROFILE
    double start, stop;
    double time_post = 0.0;
#endif

    for( auto p_it = projections_.begin(); p_it != projections_.end(); p_it++)
    {
    #ifdef ANNAR_PROFILE
        #pragma omp master
        {
            start = omp_get_wtime();
        }
    #endif
        static_cast<SpikeProjection*>(*p_it)->evaluatePostEvents();

    #ifdef ANNAR_PROFILE
        #pragma omp master
        {
            stop = omp_get_wtime();
            time_post += (stop-start) * 1000.0;
        }
        #pragma omp barrier
    #endif
    }

#ifdef ANNAR_PROFILE
    #pragma omp master
    {
        Profile::profileInstance()->appendTimePostEvent(name_, time_post);
    }
#endif
}
