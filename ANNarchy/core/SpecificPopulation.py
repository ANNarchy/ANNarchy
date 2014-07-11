from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global
import numpy as np

class PoissonPopulation(Population):
    """ 
    Population of spiking neurons following a Poisson distribution.

    Each neuron of the population will randomly emit spikes, with a mean firing rate defined by the *rates* argument.

    The mean firing rate in Hz can be a fixed value for all neurons::

        pop = PoissonPopulation(geometry=100, rates=100.0)

    but it can be modified later as a normal parameter::

        pop.rates = np.linspace(10, 150, 100)

    It is also possible to define a temporal equation for the rates, by passing a string to the argument::

        pop = PoissonPopulation(geometry=100, rates="100.0 * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0")

    The syntax of this equation follows the same structure as neural variables.

    It is also possible to add parameters to the population which can be used in the equation of *rates*::

        pop = PoissonPopulation( 
            geometry=100, 
            parameters = '''
                amp = 100.0
                frequency = 1.0
            ''',
            rates="amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0"
        )

    .. note::

        The preceding definition is fully equivalent to the definition of this neuron::

            poisson = Neuron(
                parameters = '''
                    amp = 100.0
                    frequency = 1.0
                ''',
                equations = '''
                    rates = amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                ''',
                spike = '''
                    p < rates
                '''
            )

    """

    def __init__(self, geometry, name=None, rates=10.0, parameters=None):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. 

            * *name*: unique name of the population (optional).

            * *rates*: mean firing rate of each neuron (default: 10.0 Hz). It can be a single value (e.g. 10.0) or an equation (as string).

            * *parameters*: additional parameters which can be used in the *rates* equation.
        """  
        if isinstance(rates, str):
            poisson_neuron = Neuron(
                parameters = """
                %(params)s
                """ % {'params': parameters if parameters else ''},
                equations = """
                rates = %(rates)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """ % {'rates': rates},
                spike = """
                    p < rates
                """
            )

        elif isinstance(rates, np.ndarray):
            poisson_neuron = Neuron(
                parameters = """
                rates = 10.0
                """,
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                    p <= rates
                """
            )
        else:
            poisson_neuron = Neuron(
                parameters = """
                rates = %(rates)s
                """ % {'rates': rates},
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                    p <= rates
                """
            )
        Population.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)
        
        if isinstance(rates, np.ndarray):
            self.rates = rates

class SpikeSourceArray(Population):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geoemtry can only be one-dimensional.

    *Parameters*:

    * **spike_times** : a list of times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise.

    * **name**: optional name for the population.
    """
    def __init__(self, spike_times, name=None):

        if not isinstance(spike_times, list):
            Global._error('in SpikeSourceArray, spike_times must be a Python list.')
            exit(0)

        if isinstance(spike_times[0], list): # several neurons
            nb_neurons = len(spike_times)
        else: # a single Neuron
            nb_neurons = 1
            spike_times = [ spike_times ]

        # Create a fake neuron just to be sure the description has the correct parameters
        neuron = Neuron(
            parameters="""
                spike_times = 0.0
            """,
            equations="",
            spike="(t - spike_times < dt) and (t >= spike_times)"
        )

        Population.__init__(self, geometry=nb_neurons, neuron=neuron, name=name)

        self.generator = SpikeSourceArrayGenerator(self)

        # Do some sorting to save C++ complexity
        times = []
        for neur_times in spike_times:
            times.append(sorted(list(set(neur_times))) ) # suppress doublons and sort

        self.description['parameters'][0]['init'] = times # for instantiate...
        self._times = times


    def __setattr__(self, name, value):
        if name == 'spike_times':
            if self.initialized:
                self.cyInstance._set_custom_spike_timings(self._times)
            else:
                object.__setattr__(self, name, value)
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'spike_times':
            if self.initialized:
                return self.cyInstance._get_custom_spike_timings()
            else:
                return object.__getattribute__(self, name)
        else:
            return Population.__getattribute__(self, name)

    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, 'py'+ self.class_name)(self.size, self._times)

        # Create the local attributes and actualize the initial values
        self._init_attributes()

from ANNarchy.generator.population.SpikePopulationGenerator import SpikePopulationGenerator
from ANNarchy.generator.population.Templates import *

class SpikeSourceArrayGenerator(object):

    def __init__(self, pop):
        self.pop = pop

        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.pop.class_name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.pop.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.pop.class_name+'.pyx'
        
        
    def generate(self, verbose):
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self.generate_header())
        with open(self.body, mode = 'w') as w_file:
            w_file.write(self.generate_body())
        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generate_pyx()) 


    def generate_header(self):
        template = \
"""#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"
#include "SpikePopulation.h"
using namespace ANNarchy_Global;

class %(class)s: public SpikePopulation
{
public:
    %(class)s(std::string name, int nbNeurons, std::vector< std::vector < DATA_TYPE > > spike_times);
    
    ~%(class)s();
    
    int getNeuronCount() { return nbNeurons_; }
    
    void localMetaStep(int neur_rank);
    
    void globalMetaStep();
    
    void globalOperations();
    
    void record();
    
    void reset();    // called by global_operations

    void reset(int rank);    // called by metaStep during refractoring phase

    // Access methods for the local variable spike_times
    std::vector< std::vector< DATA_TYPE > > get_spike_times() { return this->spike_times_; }
    void set_spike_times(std::vector< std::vector< DATA_TYPE> > spike_times) { this->spike_times_ = spike_times; }
    
    std::vector< DATA_TYPE > get_single_spike_times(int rank) { return this->spike_times_[rank]; }
    void set_single_spike_times(int rank, std::vector< DATA_TYPE > spike_times) { this->spike_times_[rank] = spike_times; }

private:
    
    // Generated by SpikeSourceArrayGenerator
    std::vector< std::vector< DATA_TYPE > > spike_times_;
    std::vector< DATA_TYPE > next_spikes_;
    std::vector< int> idx_next_spikes_;
};
#endif
""" % {'class' : self.pop.class_name}
    
        return template

    def generate_body(self):

        template = """#include "%(class)s.h"
#include "Global.h"
#include "SpikeDendrite.h"
#include "SpikeProjection.h"

%(class)s::%(class)s(std::string name, int nbNeurons, std::vector< std::vector < DATA_TYPE > > spike_times): SpikePopulation(name, nbNeurons)
{
    rank_ = %(pop_id)s;
    
#ifdef _DEBUG
    std::cout << name << ": %(class)s::%(class)s called (using rank " << rank_ << ")" << std::endl;
#endif

    // dt : integration step
    dt_ = 0.1;

    // Generated by SpikeSourceArrayGenerator
    spike_times_ = spike_times;
    idx_next_spikes_ = std::vector< int >(nbNeurons_, 0);
    next_spikes_ = std::vector< DATA_TYPE >(nbNeurons_, -100.0);
    for(int i = 0; i< nbNeurons_; i++){
        if(spike_times_[i].size() > 0)
           next_spikes_[i] = spike_times_[i][0];
    }

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

%(class)s::~%(class)s() 
{
#ifdef _DEBUG
    std::cout << "%(class)s::Destructor" << std::endl;
#endif
}

void %(class)s::localMetaStep(int i) 
{
    // Generated from SpikeSourceArrayGenerator
    if( DATA_TYPE(ANNarchy_Global::time)*dt_ >= next_spikes_[i] && DATA_TYPE(ANNarchy_Global::time)*dt_ - next_spikes_[i] < dt_ )
    {
        emit_spike(i);
        idx_next_spikes_[i]++;
        if(idx_next_spikes_[i] < spike_times_[i].size())
            next_spikes_[i] = spike_times_[i][idx_next_spikes_[i]];
    }    
}

void %(class)s::globalMetaStep() 
{
    spiked_ = std::vector<bool>(nbNeurons_, false); 
}

void %(class)s::globalOperations() 
{
}

void %(class)s::record() 
{
}

void %(class)s::reset() 
{    
}

void %(class)s::reset(int rank)
{
}

""" % {'class' : self.pop.class_name, 'pop_id': self.pop._id}
    
        return template


    def generate_pyx(self):
        template = \
"""from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "../build/%(class_name)s.h":
    cdef cppclass %(class_name)s:
        %(class_name)s(string name, int N, vector[ vector[double] ] spike_timings)

        int getNeuronCount()
        
        string getName()
        
        vector[ vector[int] ] get_spike_timings()        
        void reset_spike_timings()
        void start_record_spike()
        void stop_record_spike()
        
        void setMaxDelay(int)
        
        # Refractory times
        void setRefractoryTimes(vector[int])        
        vector[int] getRefractoryTimes()

        # Generated by SpikeSourceArrayGenerator
        vector[ vector[double] ] get_spike_times()  
        void set_spike_times(vector[ vector[double] ] )  

cdef class py%(class_name)s:

    cdef %(class_name)s* cInstance

    def __cinit__(self, int size, spike_timings):
        self.cInstance = new %(class_name)s('%(name)s', size, spike_timings)

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

    def set_max_delay(self, delay):
        self.cInstance.setMaxDelay(delay)

    property size:
        def __get__(self):
            return self.cInstance.getNeuronCount()
        def __set__(self, value):
            print "py%(name)s.size is a read-only attribute."

    cpdef np.ndarray _get_refractory(self):
        return np.array(self.cInstance.getRefractoryTimes())
        
    cpdef _set_refractory(self, np.ndarray value):
        self.cInstance.setRefractoryTimes(value)

    cpdef _set_custom_spike_timings(self, value):
        cdef vector[vector[double]] timings
        cdef vector[double] tmp
        for val in value:
            tmp = val
            timings.push_back(tmp)
        self.cInstance.set_spike_times(timings)

    cpdef _get_custom_spike_timings(self):
        return self.cInstance.get_spike_times()
            
"""% {'class_name' : self.pop.class_name, 'name' : self.pop.name}
    
        return template

