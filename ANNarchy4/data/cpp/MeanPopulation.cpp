#include "Global.h"
#include "MeanPopulation.h"

MeanPopulation::MeanPopulation(std::string name, int nbNeurons) : Population(name, nbNeurons, true)
{
    rate_ = std::vector<DATA_TYPE>(nbNeurons_, 0.0);
    delayedRates_ = std::deque< std::vector<DATA_TYPE> >();
}

MeanPopulation::~MeanPopulation()
{

}

std::vector<DATA_TYPE>* MeanPopulation::getRates()
{
    return &rate_;
}

std::vector<DATA_TYPE>* MeanPopulation::getRates(int delay)
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

std::vector<DATA_TYPE> MeanPopulation::getRates(std::vector<int> delays, std::vector<int> ranks)
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

void MeanPopulation::setMaxDelay(int delay)
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

DATA_TYPE MeanPopulation::sum(int neur, int typeID) {
    DATA_TYPE sum=0.0;

    /*
    for(int i=0; i< projections_[neur].size(); i++)
        if(projections_[neur][i]->getTarget() == typeID)
            sum += projections_[neur][i]->getSum();
    */

    auto it = typedProjections_[neur][typeID].begin();
    int end = typedProjections_[neur][typeID].size();
    for(int i=0; i != end; i++ )
        sum += (*(it++))->getSum();

    return sum;
}

void MeanPopulation::metaSum()
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
}

void MeanPopulation::metaStep()
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

void MeanPopulation::metaLearn()
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
