"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.SpecificPopulation import SpecificPopulation
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron



class TimedPoissonPopulation(SpecificPopulation):
    """
    Poisson population whose rate vary with the provided schedule.

    Example:

    ```python
    inp = net.create(
        TimedPoissonPopulation(
            geometry = 100,
            rates = [10., 20., 100., 20., 5.],
            schedule = [0., 100., 200., 500., 600.],
        )
    )
    ```

    This creates a population of 100 Poisson neurons whose rate will be:
    
    * 10 Hz during the first 100 ms.
    * 20 HZ during the next 100 ms.
    * 100 Hz during the next 300 ms.
    * 20 Hz during the next 100 ms.
    * 5 Hz until the end of the simulation.
    
    
    If you want the TimedPoissonPopulation to "loop" over the schedule, you can specify a period:

    ```python
    inp = net.create(
        TimedPoissonPopulation(
            geometry = 100,
            rates = [10., 20., 100., 20., 5.],
            schedule = [0., 100., 200., 500., 600.],
            period = 1000.,
        )
    )
    ```

    Here the rate will become 10Hz again every 1 second of simulation. If the period is smaller than the schedule, the remaining rates will not be set.

    You can use the `reset()` method to manually reinitialize the schedule, times becoming relative to that call:

    ```python
    net.simulate(1200.) # Should switch to 100 Hz due to the period of 1000.
    inp.reset()
    net.simulate(1000.) # Starts at 10 Hz again.
    ```

    Note that the rates are reset to the value they had before compile().

    The rates are here common to all neurons of the population. If you want each neuron to have a different rate, `rates` must have additional dimensions corresponding to the geometry of the population. The first dimension still corresponds to the schedule.

    ```python
    inp = net.create(
        TimedPoissonPopulation(
            geometry = 100,
            rates = [ 
                [10. + 0.05*i for i in range(100)], # First 100 ms
                [20. + 0.05*i for i in range(100)], # After 100 ms
            ],
            schedule = [0., 100.],
            period = 1000.,
        )
    )
    ```

    :param rates: array of firing rates (list of floats or lists of numpy arrays). The first axis corresponds to the times where the firing rate should change and have the same length as `schedule`, if used. The other dimensions must match the geometry of the population.
    :param schedule: list of times (in ms) where the firing rate should change.
    :param period: time when the timed array will be reset and start again, allowing cycling over the schedule. Default: no cycling (-1).
    :param name: optional name for the population.
    """
    def __init__(self, geometry, rates, schedule, period= -1., name=None, copied=False, net_id=0):
        
        neuron = Neuron(
            parameters = """
            proba = 1.0
            """,
            equations = """
            p = Uniform(0.0, 1.0) * 1000.0 / dt
            """,
            spike = """
            p < proba
            """,
            name="TimedPoisson",
            description="Spiking neuron following a Poisson distribution."
        )

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied, net_id=net_id)

        # Check arguments
        try:
            rates = list(rates)
        except:
            Messages._error("TimedPoissonPopulation: the rates argument must be a list of lists.")

        schedule = list(schedule)

        nb_schedules = len(rates)
        if nb_schedules != len(schedule):
            Messages._error("TimedPoissonPopulation: the first axis of the rates argument must be the same length as schedule.")


        if isinstance(rates[0], (float, int, )) : # One rate for the whole population
            rates = [np.full(self.size, rates[i]) for i in range(nb_schedules)]

        # Initial values
        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['period'] = period

    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return TimedPoissonPopulation(self.geometry, self.init['rates'] , self.init['schedule'], self.init['period'], self.name, copied=True, net_id=self.net_id if net_id is None else net_id)

    def _generate_st(self):
        """
        adjust code templates for the specific population for single thread.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_rates(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_rates() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['update_variables'] = """
        if(_active){
            //std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;

            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                proba = _buffer[_block];
                // Move to the next block
                _block++;
                // If was the last block, go back to the first block
                if (_block == _schedule.size()){
                    _block = 0;
                }
            }

            // If the timedarray is periodic, check if we arrive at that point
            if(_period > -1 && (_t == _period-1)){
                // Reset the counters
                _block=0;
                _t = -1;
            }

            // Always increment the internal time
            _t++;
        }

        if( _active ) {
            spiked.clear();

            // Updating local variables
            %(float_prec)s step = 1000.0/dt;

            #pragma omp simd
            for(int i = 0; i < size; i++){

                // p = Uniform(0.0, 1.0) * 1000.0 / dt
                p[i] = step*rand_0[i];


            }
        } // active
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        

        self._specific_template['wrapper'] = f"""
    // TimedPoissonPopulation
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)
		.def_rw("p", &PopStruct{self.id}::p)
		.def_rw("proba", &PopStruct{self.id}::proba)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_rates)
        .def("get_rates", &PopStruct{self.id}::get_rates)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods
		.def("compute_firing_rate", &PopStruct{self.id}::compute_firing_rate)

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);
""" 

    def _generate_omp(self):
        """
        adjust code templates for the specific population for openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_rates(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_rates() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['update_variables'] = """
        if(_active){
            #pragma omp single
            {
                //std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;

                // Check if it is time to set the input
                if(_t == _schedule[_block]){
                    // Set the data
                    proba = _buffer[_block];
                    // Move to the next block
                    _block++;
                    // If was the last block, go back to the first block
                    if (_block == _schedule.size()){
                        _block = 0;
                    }
                }

                // If the timedarray is periodic, check if we arrive at that point
                if(_period > -1 && (_t == _period-1)){
                    // Reset the counters
                    _block=0;
                    _t = -1;
                }

                // Always increment the internal time
                _t++;
            }
        }

        if( _active ) {
            spiked.clear();

            // Updating local variables
            %(float_prec)s step = 1000.0/dt;

            #pragma omp for simd
            for(int i = 0; i < size; i++){

                // p = Uniform(0.0, 1.0) * 1000.0 / dt
                p[i] = step*rand_0[i];


            }
        } // active
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        

        self._specific_template['wrapper'] = f"""
    // TimedPoissonPopulation
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)
		.def_rw("p", &PopStruct{self.id}::p)
		.def_rw("proba", &PopStruct{self.id}::proba)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_rates)
        .def("get_rates", &PopStruct{self.id}::get_rates)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods
		.def("compute_firing_rate", &PopStruct{self.id}::compute_firing_rate)

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);
"""

    def _generate_cuda(self):
        """
        Code generation if the CUDA paradigm is set.
        """
        # I suppress the code generation for allocating the variable r on gpu, as
        # well as memory transfer codes. This is only possible as no other variables
        # allowed in TimedArray.
        self._specific_template['init_parameters_variables'] = """
        // Random numbers
        cudaMalloc((void**)&gpu_rand_0, size * sizeof(curandState));
        init_curand_states( size, gpu_rand_0, global_seed );
"""
        self._specific_template['host_device_transfer'] = ""
        self._specific_template['device_host_transfer'] = ""

        #
        # Code for handling the buffer and schedule parameters
        self._specific_template['declare_additional'] = """
    // Custom local parameter timed array
    std::vector< int > _schedule;
    std::vector< %(float_prec)s* > gpu_buffer;
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_rates(std::vector< std::vector< %(float_prec)s > > buffer) {
        if ( gpu_buffer.empty() ) {
            gpu_buffer = std::vector< %(float_prec)s* >(buffer.size(), nullptr);
            // allocate gpu arrays
            for(int i = 0; i < buffer.size(); i++) {
                cudaMalloc((void**)&gpu_buffer[i], buffer[i].size()*sizeof(%(float_prec)s));
            }
        }

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for( ; host_it < buffer.end(); host_it++, dev_it++ ) {
            cudaMemcpy( *dev_it, host_it->data(), host_it->size()*sizeof(%(float_prec)s), cudaMemcpyHostToDevice);
        }

        gpu_proba = gpu_buffer[0];
    }

    std::vector< std::vector< %(float_prec)s > > get_rates() {
        std::vector< std::vector< %(float_prec)s > > buffer = std::vector< std::vector< %(float_prec)s > >( gpu_buffer.size(), std::vector<%(float_prec)s>(size,0.0) );

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for( ; host_it < buffer.end(); host_it++, dev_it++ ) {
            cudaMemcpy( host_it->data(), *dev_it, size*sizeof(%(float_prec)s), cudaMemcpyDeviceToHost );
        }

        return buffer;
    }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['init_additional'] = """
        // counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] = """
        // counters
        _t = 0;
        _block = 0;
        gpu_proba = gpu_buffer[0];
"""

        self._specific_template['update_variables'] = """
        if(_active) {
            // std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                gpu_proba = gpu_buffer[_block];
                // Move to the next block
                _block++;
                // If was the last block, go back to the first block
                if ( _block == _schedule.size() ) {
                    _block = 0;
                }
            }

            // If the timedarray is periodic, check if we arrive at that point
            if( (_period > -1) && (_t == _period-1) ) {
                // Reset the counters
                _block=0;
                _t = -1;
            }

            // Always increment the internal time
            _t++;
        }
"""

        self._specific_template['update_variable_body'] = """
__global__ void cuPop%(id)s_local_step( const long int t, const double dt, curandState* rand_0, double* proba, unsigned int* num_events, int* spiked, long int* last_spike )
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    %(float_prec)s step = 1000.0/dt;

    while ( i < %(size)s )
    {
        // p = Uniform(0.0, 1.0) * 1000.0 / dt
        %(float_prec)s p = curand_uniform_double( &rand_0[i] ) * step;

        if (p < proba[i]) {
            int pos = atomicAdd ( num_events, 1);
            spiked[pos] = i;
            last_spike[i] = t;
        }

        i += blockDim.x;
    }

    __syncthreads();
}
""" % {
    'id': self.id,
    'size': self.size,
    'float_prec': ConfigManager().get('precision', self.net_id)
}

        self._specific_template['update_variable_header'] = "__global__ void cuPop%(id)s_local_step( const long int t, const double dt, curandState* rand_0, double* proba, unsigned int* num_events, int* spiked, long int* last_spike );" % {'id': self.id}
        # Please notice, that the GPU kernels can be launched only with one block. Otherwise, the
        # atomicAdd which is called inside the kernel is not working correct (HD: April 1st, 2021)
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s->update();

    // Reset old events
    clear_num_events<<< 1, 1, 0, pop%(id)s->stream >>>(pop%(id)s->gpu_spike_count);
#ifdef _DEBUG
    cudaError_t err_clear_num_events_%(id)s = cudaGetLastError();
    if(err_clear_num_events_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_clear_num_events_%(id)s) << std::endl;
#endif

    // Compute current events
    cuPop%(id)s_local_step<<< 1, pop%(id)s->_threads_per_block, 0, pop%(id)s->stream >>>(
        t, dt,
        pop%(id)s->gpu_rand_0,
        pop%(id)s->gpu_proba,
        pop%(id)s->gpu_spike_count,
        pop%(id)s->gpu_spiked,
        pop%(id)s->gpu_last_spike
    );
#ifdef _DEBUG
    cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
    if(err_pop_spike_gather_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
#endif

    // transfer back the spike counter (needed by record)
    cudaMemcpyAsync( &pop%(id)s->spike_count, pop%(id)s->gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, pop%(id)s->stream );
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike_count: " << cudaGetErrorString(err) << std::endl;
#endif

    // transfer back the spiked array (needed by record)
    cudaMemcpyAsync( pop%(id)s->spiked.data(), pop%(id)s->gpu_spiked, pop%(id)s->spike_count*sizeof(int), cudaMemcpyDeviceToHost, pop%(id)s->stream );
#ifdef _DEBUG
    err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike: " << cudaGetErrorString(err) << std::endl;
#endif

""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = "//TODO: "



        self._specific_template['wrapper'] = f"""
    // TimedPoissonPopulation
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)
		.def_rw("p", &PopStruct{self.id}::p)
		.def_rw("proba", &PopStruct{self.id}::proba)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_rates)
        .def("get_rates", &PopStruct{self.id}::get_rates)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods
		.def("compute_firing_rate", &PopStruct{self.id}::compute_firing_rate)

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);
""" 

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                self.cyInstance.set_schedule( [int(val / ConfigManager().get('dt', self.net_id))  for val in value])
            else:
                self.init['schedule'] = value
        elif name == 'rates':
            if self.initialized:

                value = list(value)
                
                if isinstance(value[0], (float, int)): # same value for each neuron, create a list of list
                    value = [ [float(value[i]) for _ in range(self.size)] for i in range(len(value)) ]
                else:
                    value = [list(np.array(value[i]).flatten()) for i in range(len(value))]
                
                self.cyInstance.set_rates(value)

            else:
                self.init['rates'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /ConfigManager().get('dt', self.net_id)))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return [float(ConfigManager().get('dt', self.net_id) * val) for val in self.cyInstance.get_schedule()]
            else:
                return self.init['schedule']
        elif name == 'rates':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_rates()
                    values = []
                    for x in range(len(self.schedule)):
                        values.append(np.reshape(flat_values[x], self.geometry))
                    return values
                else:
                    return self.cyInstance.get_rates()
            else:
                return self.init['rates']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * ConfigManager().get('dt', self.net_id)
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)