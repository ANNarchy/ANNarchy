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