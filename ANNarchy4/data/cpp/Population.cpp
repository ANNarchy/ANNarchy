/*
 *    Population.cpp
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

#include "Population.h"
#include <exception>
#include <typeinfo>

Population::Population( std::string name, int nbNeurons, bool isRateType )
{
    name_ = name;
    nbNeurons_ = nbNeurons;
    dt_ = 1.0;
    maxDelay_ = 0;
    isRateType_ = isRateType;

    projections_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());
    typedProjections_ = std::vector< std::vector< std::vector<class Projection*> > >(nbNeurons_, std::vector< std::vector<class Projection*> >());

#ifdef ANNAR_PROFILE
    try{
        Profile::profileInstance()->addLayer(name_);
    }catch(std::exception e){
        std::cout << "Can't attach population to profile instance." << std::endl;
        std::cout << e.what() << std::endl;
    }
#endif

}

Population::~Population()
{

}

void Population::addProjection(int postRankID, Projection* proj)
{
#ifdef _DEBUG
    std::cout << name_ << ": added projection to neuron " << postRankID << std::endl;
#endif
    try
    {
        projections_.at(postRankID).push_back(proj);

        while(typedProjections_.at(postRankID).size() <= proj->getTarget())
        {
            typedProjections_.at(postRankID).push_back(std::vector<Projection*>());
        }

        typedProjections_.at(postRankID).at(proj->getTarget()).push_back(proj);
    }
    catch (std::exception &e)
    {
        std::cout << std::endl;
        std::cout << "Caught: " << e.what( ) << std::endl;
        std::cout << "caused by: attach a projection to neuron " << postRankID <<" but there only " << nbNeurons_ << " neurons" << std::endl;
        std::cout << std::endl;
    };
}

void Population::removeProjection(Population* pre)
{
    for(int n=0; n<nbNeurons_; n++)
    {
        for(int p=0; p< (int)projections_[n].size();p++)
        {
            if(projections_[n][p]->getPrePopulation() == pre)
                projections_[n].erase(projections_[n].begin()+p);
        }
    }
}

Projection* Population::getProjection(int neuron, int type, Population *pre)
{
    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if ( (projections_[neuron][p]->getTarget() == type) &&
                 (projections_[neuron][p]->getPrePopulation() == pre) )
            {
                return projections_[neuron][p];
            }
        }
    }

    return NULL;
}

std::vector<Projection*> Population::getProjections(int neuron, int type)
{
    std::vector<Projection*> vec = std::vector<Projection*>();

    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if(projections_[neuron][p]->getTarget() == type)
            {
                vec.push_back(projections_[neuron][p]);
            }
        }
    }

    return vec;
}



/*TODEL
 *
 * Old implementation of population class containing both mean and spike coding.
 *
Population::Population(std::string name, int nbNeurons) {
	name_ = std::move(name);

	nbNeurons_ = nbNeurons;
	rate_ = std::vector<DATA_TYPE>(nbNeurons_, 0.0);
	projections_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());
	spikeTargets_ = std::vector<std::vector<Projection*> >(nbNeurons_, std::vector<Projection*>());

	maxDelay_ = 0;
	dt_ = 1.0;
	delayedRates_ = std::deque< std::vector<DATA_TYPE> >();

	spiked_ = std::vector< bool >(nbNeurons_, false);
	spike_timings_ = std::vector< std::vector<int> >(nbNeurons_, std::vector<int>() );

#ifdef ANNAR_PROFILE
    try{
        Profile::profileInstance()->addLayer(name_);
    }catch(std::exception e){
        std::cout << "Can't attach population to profile instance." << std::endl;
        std::cout << e.what() << std::endl;
    }
#endif

#ifdef ANNAR_SCHEDULE
    // initialization of coreCounter, which counts the runtime of each thread on each CPU
    // and the number of switches of a thread between the CPUs
    coreCounter = new int* volatile [omp_get_max_threads()];
#pragma omp parallel
    {
        coreCounter[omp_get_thread_num()] = new int[omp_get_num_procs() + 2];
        for (int j = 0; j < omp_get_num_procs(); ++j) {
            coreCounter[omp_get_thread_num()][j] = 0;
        }
        coreCounter[omp_get_thread_num()][omp_get_num_procs()] = -1;
        coreCounter[omp_get_thread_num()][omp_get_num_procs() + 1] = -1;
    }
#endif
}

Population::~Population() {
    std::cout << "Population::Destructor" << std::endl;

    rate_.erase(rate_.begin(), rate_.end());
    for(auto it=delayedRates_.begin(); it<delayedRates_.end(); it++)
        (*it).erase((*it).begin(), (*it).end());

    for(int n=0; n<nbNeurons_; n++) {
        while(!projections_[n].empty()){
            delete projections_[n].back();
            projections_[n].pop_back();
        }
        //projections_[n].erase(projections_[n].begin(), projections_[n].end());
    }

#ifdef ANNAR_PROFILE
    if(cs)
        fclose(cs);
    if(gl)
        fclose(gl);
    if(ll)
        fclose(ll);
#endif

#ifdef ANNAR_SCHEDULE
    // delete the coreCounter
    for (int i = 0; i < omp_get_max_threads(); i++) {
        delete[] coreCounter[i];
    }
    delete[] coreCounter;
#endif
}

std::vector<Projection*> Population::getProjections(int neuron, int type) {
    std::vector<Projection*> vec = std::vector<Projection*>();

    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if(projections_[neuron][p]->getTarget() == type)
            {
                vec.push_back(projections_[neuron][p]);
            }
        }
    }

    return vec;
}

Projection* Population::getProjection(int neuron, int type, Population *pre) {
    if (neuron < projections_.size())
    {
        for(int p=0; p< projections_[neuron].size(); p++)
        {
            if ( (projections_[neuron][p]->getTarget() == type) &&
                 (projections_[neuron][p]->getPrePopulation() == pre) )
            {
                return projections_[neuron][p];
            }
        }
    }

    return NULL;
}

void Population::printRates() {
	for(int n=0; n<nbNeurons_; n++) {
		printf("%.02f ", rate_[n]);
		if((n>0)&&(n%10==0))
			printf("\n");
	}
	printf("\n");
}

DATA_TYPE Population::sum(int neur, int typeID) {
	DATA_TYPE sum=0.0;

	for(int i=0; i< projections_[neur].size(); i++)
		if(projections_[neur][i]->getTarget() == typeID)
			sum += projections_[neur][i]->getSum();

	return sum;
}

std::vector<DATA_TYPE> Population::getRates(std::vector<int> delays, std::vector<int> ranks) {
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

void Population::setMaxDelay(int delay) {
	// TODO:
	// maybe we should take the current fire rate as initial value
#ifdef _DEBUG
    std::cout << "Population " << name_ << " got new max delay: " << delay << std::endl;
#endif
	if(delay > maxDelay_) {
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

void Population::addProjection(int postRankID, Projection* proj) {
#ifdef _DEBUG
	std::cout << name_ << ": added projection to neuron " << postRankID << std::endl;
#endif
	try
	{
		projections_.at(postRankID).push_back(proj);
	}
	catch (std::exception &e)
	{
		std::cout << std::endl;
		std::cout << "Caught: " << e.what( ) << std::endl;
		std::cout << "Tried to attach projection to neuron " << postRankID <<" but there only " << nbNeurons_ << " neurons" << std::endl;
		std::cout << std::endl;
	};
}

void Population::addSpikeTarget(Projection* proj)
{
#ifdef _DEBUG
    std::cout << name_ << ": added projection as spike target " << std::endl;
#endif
    for(unsigned int n=0; n< nbNeurons_; n++)
    {
        spikeTargets_.at(n).push_back(proj);
    }
}

void Population::removeProjection(Population* pre) {
	for(int n=0; n<nbNeurons_; n++) {
		for(int p=0; p< (int)projections_[n].size();p++) {
			if(projections_[n][p]->getPrePopulation() == pre)
				projections_[n].erase(projections_[n].begin()+p);
		}
	}
}

void Population::metaSum() {

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
        if ( projections_[n].size() > 0 && omp_get_thread_num() == 0 )
            std::cout << name_<<"("<< n << "): "<< projections_[n].size()<< " projections."<< std::endl;
    #endif
        for(int p=0; p< (int)projections_[n].size();p++)
        {
        #ifdef _DEBUG
            std::cout << "reference: " << projections_[n][p] << std::endl;
            std::cout << "\tpost = " << name_ << std::endl;
            std::cout << "\tpre = " << projections_[n][p]->getPrePopulation()->getName() << std::endl;
            std::cout << "\tsynaseCount = " << (int)(projections_[n][p]->getSynapseCount()) << std::endl;
        #endif
            projections_[n][p]->computeSum();
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

#ifdef ANNAR_SCHEDULE
    // output the coreCounter
    if(ANNarchy_Global::time % 1000 == 0) {
        printf("\n'%s' - time: %d\n", name_.c_str(), ANNarchy_Global::time);
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            for (int j = 0; j < omp_get_num_procs(); ++j) {
                printf("%d\t", coreCounter[i][j]);
            }
            printf("(%d)\n", coreCounter[i][omp_get_num_procs()]);
        }
    }
#endif

}

void Population::metaStep() {
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

//
// projection update for post neuron based variables
void Population::metaLearn()
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

void Population::globalOperations() {

}
*/
