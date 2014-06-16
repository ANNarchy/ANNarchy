##############################################
### Spike2Rate
##############################################

s2r_header_template = """
#ifndef __ANNarchy_%(class_name)s_H__
#define __ANNarchy_%(class_name)s_H__

#include "Global.h"
#include "RatePopulation.h"
#include "%(pre_population)s.h"
using namespace ANNarchy_Global;

class %(class_name)s: public RatePopulation
{
public:
    %(class_name)s(std::string name, int nbNeurons, int id_pre);
    
    ~%(class_name)s();
    
    void prepareNeurons();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

    // Access methods for the local variable r
    std::vector<DATA_TYPE> get_r() { return this->r_; }
    void set_r(std::vector<DATA_TYPE> r) { this->r_ = r; }
    
    DATA_TYPE get_single_r(int rank) { return this->r_[rank]; }
    void set_single_r(int rank, DATA_TYPE r) { this->r_[rank] = r; }

    std::vector< std::vector< DATA_TYPE > > get_recorded_r() { return this->recorded_r_; }                    
    void start_record_r() { this->record_r_ = true; }
    void stop_record_r() { this->record_r_ = false; }
    void clear_recorded_r() { this->recorded_r_.clear(); }

    // Access methods for the parameter scaling_
    DATA_TYPE get_scaling() { return scaling_; }
    void set_scaling(DATA_TYPE value) { scaling_ = value; }

    // Access methods for the parameter window_
    DATA_TYPE get_window() { return window_; }
    void set_window(DATA_TYPE value) { window_ = value; }

    // Access methods for the parameter smooth_
    DATA_TYPE get_smooth() { return smooth_; }
    void set_smooth(DATA_TYPE value) { smooth_ = value; }

private:

    // Population to convert
    %(pre_population)s* pre_population_;

    // Transformation
    DATA_TYPE scaling_;
    DATA_TYPE window_;
    DATA_TYPE smooth_;
    std::vector< std::deque<int> > last_spikes_;
    int nbSpikesInTheLast(int rank, int t);

    // r_ : local
    bool record_r_; 
    std::vector< std::vector<DATA_TYPE> > recorded_r_;    
};
#endif
"""

s2r_body_template = """
#include "%(class_name)s.h"
#include "Global.h"

%(class_name)s::%(class_name)s(std::string name, int nbNeurons, int id_pre): RatePopulation(name, nbNeurons)
{
    rank_ = %(id)s;
    pre_population_ = static_cast<class %(pre_population)s*>(Network::instance()->getPopulation(id_pre));
    
#ifdef _DEBUG
    std::cout << name << ": %(class_name)s::%(class_name)s called (using rank " << rank_ << ")" << std::endl;
#endif

    // transformations
    scaling_ = 1.0; // to which rate corresponds 1 Hz
    smooth_ = 1.0; // time constant of the low.pass filter applied to the rates
    window_ = 100.0; // sliding window to retreve the last spikes
    last_spikes_ = std::vector< std::deque<int> > (nbNeurons_, std::deque<int>(1, -10000));

    // r_ : local
    r_ = std::vector<DATA_TYPE> (nbNeurons_, 0.0);
    record_r_ = false; 
    recorded_r_ = std::vector< std::vector< DATA_TYPE > >();    

    // dt : integration step
    dt_ = 1.0;


    try
    {
        Network::instance()->addPopulation(this);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to attach population"<< std::endl;
        std::cout << e.what() << std::endl;
    }
}

%(class_name)s::~%(class_name)s() 
{
#ifdef _DEBUG
    std::cout << "%(class_name)s::Destructor" << std::endl;
#endif

    r_.clear();
}

void %(class_name)s::prepareNeurons() 
{

    if (maxDelay_ > dt_)
    {
    #ifdef _DEBUG_DELAY
        std::cout << name_ << ": delay = " << maxDelay_ << std::endl;
        std::cout << "OLD ( t = "<< ANNarchy_Global::time << ")" << std::endl;
        for ( int i = 0; i < delayedRates_.size(); i++)
        {
            std::cout << i <<": ";
            for ( auto it = delayedRates_[i].begin(); it != delayedRates_[i].end(); it++)
                std::cout << *it << " ";
            std::cout << std::endl;            
        }
    #endif
    
        delayedRates_.push_front(r_);
        delayedRates_.pop_back();
        
    #ifdef _DEBUG_DELAY
        std::cout << "NEW ( t = "<< ANNarchy_Global::time << ")" << std::endl;
        for ( int i = 0; i < delayedRates_.size(); i++)
        {
            std::cout << i <<": ";
            for ( auto it = delayedRates_[i].begin(); it != delayedRates_[i].end(); it++)
                std::cout << *it << " ";
            std::cout << std::endl;            
        }
    #endif
    }

}

void %(class_name)s::resetToInit() 
{

    // transformations
    scaling_ = 1.0; // to which rate corresponds 1 Hz
    smooth_ = 1.0; // time constant of the low.pass filter applied to the rates
    window_ = 100.0; // sliding window to retreve the last spikes
    last_spikes_ = std::vector< std::deque<int> > (nbNeurons_, std::deque<int>(1, -10000));

    // r_ : local
    r_ = std::vector<DATA_TYPE> (nbNeurons_, 0.0);   

}


int %(class_name)s::nbSpikesInTheLast(int rank, int t) 
{ 
    int nb = 0;
    for(int i=0; i < last_spikes_[rank].size(); i++){
        if(last_spikes_[rank][i] > ANNarchy_Global::time - t){
            nb++;
        }
        else{
            last_spikes_[rank].erase(last_spikes_[rank].begin()+i);
        }
    }
    return nb;

}

void %(class_name)s::localMetaStep(int i) 
{

    // Increase when spiking
    if (pre_population_->hasSpiked(i, ANNarchy_Global::time-1)){
       last_spikes_[i].push_front(ANNarchy_Global::time -1);
    }

    r_[i] += dt_ * ( scaling_*1000.0 / window_ * DATA_TYPE(this->nbSpikesInTheLast(i, int(window_*dt_))) - r_[i] ) /smooth_;


}

void %(class_name)s::globalMetaStep() 
{
                
}

void %(class_name)s::globalOperations() 
{

}

void %(class_name)s::record() 
{
    if(record_r_)
        recorded_r_.push_back(r_);
}
"""

s2r_pyx_template = """
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N, int id_pre)
        int getNeuronCount()
        string getName()
        void resetToInit()
        void setMaxDelay(int)

        # Local r
        vector[double] get_r()
        void set_r(vector[double] values)
        double get_single_r(int rank)
        void set_single_r(int rank, double values)
        void start_record_r()
        void stop_record_r()
        void clear_recorded_r()
        vector[vector[double]] get_recorded_r()

        # Global window_
        double get_window()
        void set_window(double value)

        # Global scaling_
        double get_scaling()
        void set_scaling(double value)

        # Global smooth_
        double get_smooth()
        void set_smooth(double value)



cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size, int id_pre):
        self.cInstance = new %(class_name)s('%(name)s', size, id_pre)

    def name(self):
        return self.cInstance.getName()

    def reset(self):
        self.cInstance.resetToInit()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "pyconverter.size is a read-only attribute."
            
    # local: r
    cpdef np.ndarray _get_r(self):
        return np.array(self.cInstance.get_r())
        
    cpdef _set_r(self, np.ndarray value):
        self.cInstance.set_r(value)
        
    cpdef double _get_single_r(self, rank):
        return self.cInstance.get_single_r(rank)

    def _set_single_r(self, int rank, double value):
        self.cInstance.set_single_r(rank, value)

    def _start_record_r(self):
        self.cInstance.start_record_r()

    def _stop_record_r(self):
        self.cInstance.stop_record_r()

    cpdef np.ndarray _get_recorded_r(self):
        tmp = np.array(self.cInstance.get_recorded_r())
        self.cInstance.clear_recorded_r()
        return tmp

    # global: window
    cpdef double _get_window(self):
        return self.cInstance.get_window()
    def _set_window(self, double value):
        self.cInstance.set_window(value)

    # global: scaling
    cpdef double _get_scaling(self):
        return self.cInstance.get_scaling()
    def _set_scaling(self, double value):
        self.cInstance.set_scaling(value)

    # global: smooth
    cpdef double _get_smooth(self):
        return self.cInstance.get_smooth()
    def _set_smooth(self, double value):
        self.cInstance.set_smooth(value)
"""

##############################################
### Rate2Spike
##############################################

r2s_header_template = """
#ifndef __ANNarchy_%(class_name)s_H__
#define __ANNarchy_%(class_name)s_H__

#include "Global.h"
#include "SpikePopulation.h"
#include "%(pre_population)s.h"
using namespace ANNarchy_Global;

class %(class_name)s: public SpikePopulation
{
public:
    %(class_name)s(std::string name, int nbNeurons, int id_pre);
    
    ~%(class_name)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void prepareNeurons();
    
    void resetToInit();
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();

    void propagateSpikes();
    
    void evaluateSpikes();
    
    void reset();    // called by global_operations

    void reset(int rank);    // called by metaStep during refractoring phase


    

    // Access methods for the local variable rates
    std::vector<double> get_rates() { return this->rates_; }
    void set_rates(std::vector<double> rates) { this->rates_ = rates; }
    
    double get_single_rates(int rank) { return this->rates_[rank]; }
    void set_single_rates(int rank, double rates) { this->rates_[rank] = rates; }

    std::vector< std::vector< double > > get_recorded_rates() { return this->recorded_rates_; }                    
    void start_record_rates() { this->record_rates_ = true; }
    void stop_record_rates() { this->record_rates_ = false; }
    void clear_recorded_rates() { this->recorded_rates_.clear(); }

    // Access methods for the local variable p
    std::vector<double> get_p() { return this->p_; }
    void set_p(std::vector<double> p) { this->p_ = p; }
    
    double get_single_p(int rank) { return this->p_[rank]; }
    void set_single_p(int rank, double p) { this->p_[rank] = p; }

    std::vector< std::vector< double > > get_recorded_p() { return this->recorded_p_; }                    
    void start_record_p() { this->record_p_ = true; }
    void stop_record_p() { this->record_p_ = false; }
    void clear_recorded_p() { this->recorded_p_.clear(); }


    // Access methods for the parameter scaling_
    DATA_TYPE get_scaling() { return scaling_; }
    void set_scaling(DATA_TYPE value) { scaling_ = value; }

private:

    // Population to convert
    %(pre_population)s* pre_population_;

    // Transformation
    DATA_TYPE scaling_;

    // rates_ : local
    std::vector<double> rates_;
    bool record_rates_; 
    std::vector< std::vector<double> > recorded_rates_;    

    // p_ : local
    std::vector<double> p_;
    bool record_p_; 
    std::vector< std::vector<double> > recorded_p_;    


    std::vector<double> __rand_0__;
    UniformDistribution<double>* __dist_0__;


    std::vector<int> propagate_;    ///< neurons which will propagate their spike
    std::vector<int> reset_;    ///< neurons which will reset after current eval
    
};
#endif
"""

r2s_body_template = """
#include "%(class_name)s.h"
#include "Global.h"
#include "SpikeDendrite.h"
#include "SpikeProjection.h"

%(class_name)s::%(class_name)s(std::string name, int nbNeurons, int id_pre): SpikePopulation(name, nbNeurons)
{
    rank_ = 0;
    pre_population_ = static_cast<class %(pre_population)s*>(Network::instance()->getPopulation(id_pre));
    
#ifdef _DEBUG
    std::cout << name << ": %(class_name)s::%(class_name)s called (using rank " << rank_ << ")" << std::endl;
#endif

    // transformations
    scaling_ = 1.0; // to which rate corresponds 1 Hz

    // rates_ : local
    rates_ = std::vector<double> (nbNeurons_, 10.0);
    record_rates_ = false; 
    recorded_rates_ = std::vector< std::vector< double > >();    

    // p_ : local
    p_ = std::vector<double> (nbNeurons_, 0.0);
    record_p_ = false; 
    recorded_p_ = std::vector< std::vector< double > >();    

    // dt : integration step
    dt_ = 1.0;

    __dist_0__ = new UniformDistribution<double>(0.0, 1.0, -1.0);

    refractory_times_ = std::vector<int>(nbNeurons, 0);
    

    spiked_ = std::vector<bool>(nbNeurons_, false);
    
    try
    {
        Network::instance()->addPopulation(this);
    }
    catch(std::exception e)
    {
        std::cout << "Failed to attach population"<< std::endl;
        std::cout << e.what() << std::endl;
    }
}

%(class_name)s::~%(class_name)s() 
{
#ifdef _DEBUG
    std::cout << "%(class_name)s::Destructor" << std::endl;
#endif

    rates_.clear();
    p_.clear();
}

void %(class_name)s::prepareNeurons() 
{


    updateRefactoryCounter();
}

void %(class_name)s::resetToInit() 
{

    // rates_ : local
    rates_ = std::vector<double> (nbNeurons_, 10.0);   

    // p_ : local
    p_ = std::vector<double> (nbNeurons_, 0.0);   

}

void %(class_name)s::localMetaStep(int i) 
{

    // p = Uniform(0.0, 1.0) * 1000.0 / dt
    p_[i] = 1000.0*__rand_0__[i]/dt_;

    if( p_[i] <= rates_[i] * scaling_ )
    {
        if (refractory_counter_[i] < 1)
        {
            #pragma omp critical
            {
                //std::cout << "emit spike (pop " << name_ <<")["<<i<<"] ( time="<< ANNarchy_Global::time<< ")" << std::endl;
                this->propagate_.push_back(i);
                this->reset_.push_back(i);
                
                lastSpike_[i] = ANNarchy_Global::time;
                if(record_spike_){
                    spike_timings_[i].push_back(ANNarchy_Global::time);
                }
                spiked_[i] = true;
            }
        }
    }
    
}

void %(class_name)s::globalMetaStep() 
{
    spiked_ = std::vector<bool>(nbNeurons_, false);
    __rand_0__ = __dist_0__->getValues(nbNeurons_);

    // rates comes from pre-population
    rates_ = pre_population_->get_r();
            
}

void %(class_name)s::globalOperations() 
{
    reset();
}

void %(class_name)s::record() 
{

    if(record_rates_)
        recorded_rates_.push_back(rates_);

    if(record_p_)
        recorded_p_.push_back(p_);

    for(unsigned int p=0; p< projections_.size(); p++)
    {
        projections_[p]->record();
    }
}

void %(class_name)s::propagateSpikes() 
{
    if (!propagate_.empty())
    {
        // emit a postsynaptic spike on receiving projections
        for( auto p_it = projections_.begin(); p_it != projections_.end(); p_it++)
        {
            if ( static_cast<SpikeProjection*>(*p_it)->isLearning() )
                static_cast<SpikeProjection*>(*p_it)->postEvent(propagate_);
        }

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
    }
}

void %(class_name)s::evaluateSpikes(){}

void %(class_name)s::reset() 
{
    if (!reset_.empty())
    {
        for (auto it = reset_.begin(); it != reset_.end(); it++)
        {


            refractory_counter_[*it] = refractory_times_[*it];
        }
        
        reset_.erase(reset_.begin(), reset_.end());
    }
    
}

void %(class_name)s::reset(int rank){}
"""

r2s_pyx_template = """
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N, int id_pre)

        int getNeuronCount()
        
        string getName()
        
        vector[ vector[int] ] get_spike_timings()        
        void reset_spike_timings()
        void start_record_spike()
        void stop_record_spike()
        
        void resetToInit()
        
        void setMaxDelay(int)
        
        void setRefractoryTimes(vector[int])
        
        vector[int] getRefractoryTimes()



        # Local rates
        vector[double] get_rates()
        void set_rates(vector[double] values)
        double get_single_rates(int rank)
        void set_single_rates(int rank, double values)
        void start_record_rates()
        void stop_record_rates()
        void clear_recorded_rates()
        vector[vector[double]] get_recorded_rates()

        # Local p
        vector[double] get_p()
        void set_p(vector[double] values)
        double get_single_p(int rank)
        void set_single_p(int rank, double values)
        void start_record_p()
        void stop_record_p()
        void clear_recorded_p()
        vector[vector[double]] get_recorded_p()

        # Global scaling_
        double get_scaling()
        void set_scaling(double value)



cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size, int id_pre):
        self.cInstance = new %(class_name)s('%(name)s', size, id_pre)

    def name(self):
        return self.cInstance.getName()

    cpdef np.ndarray _get_recorded_spike(self):
        cdef np.ndarray tmp
        tmp = np.array( self.cInstance.get_spike_timings() )
        self.cInstance.reset_spike_timings()
        return tmp

    def _start_record_spike(self):
        self.cInstance.start_record_spike()

    def _stop_record_spike(self):
        self.cInstance.stop_record_spike()

    def reset(self):
        self.cInstance.resetToInit()

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "pyspiking.size is a read-only attribute."

    cpdef np.ndarray _get_refractory_times(self):
        return np.array(self.cInstance.getRefractoryTimes())
        
    cpdef _set_refractory_times(self, np.ndarray value):
        self.cInstance.setRefractoryTimes(value)
            


    # local: rates
    cpdef np.ndarray _get_rates(self):
        return np.array(self.cInstance.get_rates())
        
    cpdef _set_rates(self, np.ndarray value):
        self.cInstance.set_rates(value)
        
    cpdef double _get_single_rates(self, rank):
        return self.cInstance.get_single_rates(rank)

    def _set_single_rates(self, int rank, double value):
        self.cInstance.set_single_rates(rank, value)

    def _start_record_rates(self):
        self.cInstance.start_record_rates()

    def _stop_record_rates(self):
        self.cInstance.stop_record_rates()

    cpdef np.ndarray _get_recorded_rates(self):
        tmp = np.array(self.cInstance.get_recorded_rates())
        self.cInstance.clear_recorded_rates()
        return tmp
        


    # local: p
    cpdef np.ndarray _get_p(self):
        return np.array(self.cInstance.get_p())
        
    cpdef _set_p(self, np.ndarray value):
        self.cInstance.set_p(value)
        
    cpdef double _get_single_p(self, rank):
        return self.cInstance.get_single_p(rank)

    def _set_single_p(self, int rank, double value):
        self.cInstance.set_single_p(rank, value)

    def _start_record_p(self):
        self.cInstance.start_record_p()

    def _stop_record_p(self):
        self.cInstance.stop_record_p()

    cpdef np.ndarray _get_recorded_p(self):
        tmp = np.array(self.cInstance.get_recorded_p())
        self.cInstance.clear_recorded_p()
        return tmp



    # global: scaling
    cpdef double _get_scaling(self):
        return self.cInstance.get_scaling()
    def _set_scaling(self, double value):
        self.cInstance.set_scaling(value)
"""