"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.SpecificPopulation import SpecificPopulation
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron

class TimedArray(SpecificPopulation):
    """
    Data structure holding sequential inputs for a rate-coded network.

    The input values are stored in the (recordable) attribute `r`, without any further processing.
    You will need to connect this population to another one using the ``connect_one_to_one()`` method.

    By default, the firing rate of this population will iterate over the different values step by step:

    ```python
    inputs = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    )

    inp = ann.TimedArray(rates=inputs)

    pop = ann.Population(10, ...)

    proj = ann.Projection(inp, pop, 'exc')
    proj.connect_one_to_one(1.0)

    ann.compile()

    ann.simulate(10.)
    ```

    This creates a population of 10 neurons whose activity will change during the first 10*dt milliseconds of the simulation. After that delay, the last input will be kept (i.e. 1 for the last neuron).

    If you want the TimedArray to "loop" over the different input vectors, you can specify a period for the inputs:

    ```python
    inp = ann.TimedArray(rates=inputs, period=10.)
    ```

    If the period is smaller than the length of the rates, the last inputs will not be set.

    If you do not want the inputs to be set at every step, but every 10 ms for example, youcan use the ``schedule`` argument:

    ```python
    inp = ann.TimedArray(rates=inputs, schedule=10.)
    ```

    The input [1, 0, 0,...] will stay for 10 ms, then[0, 1, 0, ...] for the next 10 ms, etc...

    If you need a less regular schedule, you can specify it as a list of times:

    ```python
    inp = ann.TimedArray(rates=inputs, schedule=[10., 20., 50., 60., 100., 110.])
    ```

    The first input is set at t = 10 ms (r = 0.0 in the first 10 ms), the second at t = 20 ms, the third at t = 50 ms, etc.

    If you specify less times than in the array of rates, the last ones will be ignored.

    Scheduling can be combined with periodic cycling. Note that you can use the ``reset()`` method to manually reinitialize the TimedArray, times becoming relative to that call:

    ```python
    ann.simulate(100.) # ten inputs are shown with a schedule of 10 ms
    inp.reset()
    ann.simulate(100.) # the same ten inputs are presented again.
    ```

    :param rates: array of firing rates. The first axis corresponds to time, the others to the desired dimensions of the population.
    :param geometry: desired dimensions of the population. This argument will be considered if *rates* is None.
    :param schedule: either a single value or a list of time points where inputs should be set. Default: every timestep.
    :param period: time when the timed array will be reset and start again, allowing cycling over the inputs. Default: no cycling (-1.).

    """
    def __init__(self, 
                 rates:np.ndarray=None, 
                 geometry:int|tuple=None, 
                 schedule:float=0., 
                 period:float=-1., 
                 name:str=None, 
                 copied:bool=False):

        # Sanity check
        if rates is None and geometry is None:
            Messages._error("TimedArray: either *rates* or *geometry* argument must be set.")

        # Geometry of the population
        if rates is not None:
            if geometry is None:
                geometry = rates.shape[1:]
            else:
                if geometry != rates.shape[1:]:
                    Messages._warning("TimedArray: mismatch between *rates* and *geometry* dimensions detected.")

        # Create input neuron type
        neuron = Neuron(
            parameters="",
            equations="r = 0.0",
            name="Timed Array",
            description="Timed array sets inputs (shape = {}) sequentially with schedule = {} and period = {}.".format(geometry, schedule, period)
        )

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied)

        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['period'] = period

        if rates is not None:
            self.update(rates=rates, period=period, schedule=schedule)

    @property
    def r(self):
        if self.initialized:
            return self._get_cython_attribute("r")
        else:
            Messages._error("Read-out of 'r' is only possible after compile.")

    @r.setter
    def r(self, new_r):
        Messages._error("The value of r is defined through the '*'rates' argument.")

    def update(self, rates, schedule=0., period=-1):
        """
        Set a new list of inputs. The first axis corresponds to time, the others to the desired dimensions of the population. Note, the
        geometry is set during construction phase of the object.

        :param rates: array of firing rates. The first axis corresponds to time, the others to the desired dimensions of the population.
        :param schedule: either a single value or a list of time points where inputs should be set. Default: every timestep.
        :param period: time when the timed array will be reset and start again, allowing cycling over the inputs. Default: no cycling (-1.).
        """
        self.rates = rates
        self.period = period

        # Check the schedule
        if isinstance(schedule, (int, float)):
            if float(schedule) <= 0.0:
                schedule = get_global_config('dt')

            self.schedule = [ float(schedule*i) for i in range(rates.shape[0])]
        else:
            self.schedule = schedule

        if len(self.schedule) > self.rates.shape[0]:
            Messages._error('TimedArray: the length of the schedule parameter cannot exceed the first dimension of the rates parameter.')

        if len(self.schedule) < self.rates.shape[0]:
            Messages._warning('TimedArray: the length of the schedule parameter is smaller than the first dimension of the rates parameter (more data than time points). Make sure it is what you expect.')

    def _copy(self):
        "Returns a copy of the population when creating networks."
        return TimedArray(rates=self.rates, geometry=self.geometry, schedule=self.schedule, period=self.period, name=self.name, copied=True)

    def _generate_st(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedArray
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedArray
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[_block]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedArray
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

        self._specific_template['update_variables'] = """
        if(_active) {
        #ifdef _DEBUG
            std::cout << "TimedArray::update() - " << _t << " " << _block<< " " << _schedule[_block] << std::endl;
        #endif

            // Check if it is time to set the input
            if (_t == _schedule[_block]) {
                // sanity check
                if (_buffer.empty()) {
                    std::cerr << "TimedArray: no data being set ..." << std::endl;
                    r = std::vector<%(float_prec)s>(size, 0.0);
                    return;
                }

                // sanity check
                if (_buffer.size() <= _block) {
                    std::cerr << "TimedArray: not enough data being set ..." << std::endl;
                    r = std::vector<%(float_prec)s>(size, 0.0);
                    return;
                }

                // Set the data
                r = _buffer[_block];
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

        #ifdef _DEBUG
            std::cout << "TimedArray::update(t="<< t <<") - current buffer (min/max) = [" << *std::min_element(r.begin(), r.end()) << "," << *std::max_element(r.begin(), r.end()) <<  "]" << std::endl;
        #endif
        }
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': get_global_config('precision')}

    def _generate_omp(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedArray
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedArray
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[_block]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedArray
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

        # HD (28th Jun. 24): contrary to the single-thread codes, where we use 'return' to escape the function
        #                    execution, OpenMP does not allow 'return'/'continue' in the execution block. Therefore,
        #                    we need to use the if-else tree.
        self._specific_template['update_variables'] = """
        if(_active){
            #pragma omp single
            {
            #ifdef _DEBUG
                std::cout << "TimedArray::update() - " << _t << " " << _block<< " " << _schedule[_block] << std::endl;
            #endif

                // Check if it is time to set the input
                if (_t == _schedule[_block]) {
                    // sanity check
                    if (_buffer.empty()) {
                        std::cerr << "TimedArray: no data being set ..." << std::endl;
                        r = std::vector<%(float_prec)s>(size, 0.0);
                    }

                    // sanity check
                    else if (_buffer.size() <= _block) {
                        std::cerr << "TimedArray: not enough data being set ..." << std::endl;
                        r = std::vector<%(float_prec)s>(size, 0.0);
                    }

                    // everything appears right, proceed
                    else {
                        // Set the data
                        r = _buffer[_block];
                        // Move to the next block
                        _block++;
                        // If was the last block, go back to the first block
                        if (_block == _schedule.size()){
                            _block = 0;
                        }
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
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': get_global_config('precision')}

    def _generate_cuda(self):
        """
        adjust code templates for the specific population for single thread and CUDA.
        """
        # HD (18. Nov 2016)
        # I suppress the code generation for allocating the variable r on gpu, as
        # well as memory transfer codes. This is only possible as no other variables
        # allowed in TimedArray.
        self._specific_template['init_parameters_variables'] = ""
        self._specific_template['host_device_transfer'] = ""
        self._specific_template['device_host_transfer'] = ""

        #
        # Code for handling the buffer and schedule parameters
        self._specific_template['declare_additional'] = """
    // Custom local parameter timed array
    std::vector< int > _schedule;
    std::vector< %(float_prec)s* > _gpu_buffer;
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) {
        if ( _gpu_buffer.empty() ) {
            // host holds a set of pointers
            _gpu_buffer = std::vector< %(float_prec)s* >(buffer.size(), nullptr);

            // allocate gpu arrays
            for(int i = 0; i < buffer.size(); i++) {
                cudaMalloc((void**)&_gpu_buffer[i], buffer[i].size()*sizeof(%(float_prec)s));
            }
        }

        auto host_it = buffer.begin();
        auto dev_it = _gpu_buffer.begin();
        for (; host_it < buffer.end(); host_it++, dev_it++) {
            cudaMemcpy( *dev_it, host_it->data(), host_it->size()*sizeof(%(float_prec)s), cudaMemcpyHostToDevice);
        }

        gpu_r = _gpu_buffer[_block];
    }
    std::vector< std::vector< %(float_prec)s > > get_buffer() {
        std::vector< std::vector< %(float_prec)s > > buffer = std::vector< std::vector< %(float_prec)s > >( _gpu_buffer.size(), std::vector<%(float_prec)s>(size,0.0) );

        auto host_it = buffer.begin();
        auto dev_it = _gpu_buffer.begin();
        for (; host_it < buffer.end(); host_it++, dev_it++) {
            cudaMemcpy( host_it->data(), *dev_it, size*sizeof(%(float_prec)s), cudaMemcpyDeviceToHost );
        }

        return buffer;
    }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': get_global_config('precision')}
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
        gpu_r = _gpu_buffer[0];
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( np.array(schedule, dtype=int) )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_periodic(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id, 'float_prec': get_global_config('precision') }

        # there is no GPU-side computation
        self._specific_template['update_variable_body'] = ""
        self._specific_template['update_variable_invoke'] = ""
        self._specific_template['update_variable_header'] = ""

        # we switch the GPU buffer which is read out in each time step
        self._specific_template['update_variables'] = """
        if(_active) {
        #ifdef _DEBUG
            std::cout << "TimedArray::update() - " << _t << " " << _block<< " " << _schedule[_block] << std::endl;
        #endif
            // Check if it is time to set the input
            if (_t == _schedule[_block]) {
                // sanity check
                if (_gpu_buffer.empty()) {
                    std::cerr << "TimedArray: no data being set ..." << std::endl;
                    gpu_r = _gpu_buffer[0];
                    return;
                }

                // sanity check
                if (_gpu_buffer.size() <= _block) {
                    std::cerr << "TimedArray: not enough data being set ..." << std::endl;
                    gpu_r = _gpu_buffer[0];
                    return;
                }

                // Set the data
                gpu_r = _gpu_buffer[_block];
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
        # call the switch of CPU-buffers (host-side)
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s.update();
""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = """
        // r
        size_in_bytes += sizeof(std::vector<%(float_prec)s>);
        size_in_bytes += r.capacity() * sizeof(%(float_prec)s);

        // schedule
        size_in_bytes += sizeof(std::vector<int>);
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // gpu_buffer
        size_in_bytes += sizeof(std::vector<%(float_prec)s*>);
        size_in_bytes += _gpu_buffer.capacity() * sizeof(%(float_prec)s*);
""" % {'float_prec': get_global_config('precision')}

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                val_int = np.array((np.atleast_1d(value) / get_global_config('dt')), dtype=np.int32)
                self.cyInstance.set_schedule( val_int )
            else:
                self.init['schedule'] = value
        elif name == 'rates':
            if self.initialized:
                if value is None:
                    return # nothing to do

                if len(value.shape) > 2:
                    if value.shape[1:] != self.geometry:
                        Messages._error("TimedArray: mismatch between *rates* argument (", value.shape[1:], ") and stored geometry (", self.geometry, ").")

                    # we need to flatten the provided data
                    flat_values = value.reshape( (value.shape[0], self.size) )
                    self.cyInstance.set_rates( flat_values )
                else:
                    self.cyInstance.set_rates( value )
            else:
                self.init['rates'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /get_global_config('dt')))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return get_global_config('dt') * self.cyInstance.get_schedule()
            else:
                return self.init['schedule']
        elif name == 'rates':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_rates()
                    values = np.zeros( tuple( [len(self.schedule)] + list(self.geometry) ) )
                    for x in range(len(self.schedule)):
                        values[x] = np.reshape( flat_values[x], self.geometry)
                    return values
                else:
                    return self.cyInstance.get_rates()
            else:
                return self.init['rates']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * get_global_config('dt')
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)

class TimedPoissonPopulation(SpecificPopulation):
    """
    Poisson population whose rate vary with the provided schedule.

    Example:

    ```python
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [10., 20., 100., 20., 5.],
        schedule = [0., 100., 200., 500., 600.],
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
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [10., 20., 100., 20., 5.],
        schedule = [0., 100., 200., 500., 600.],
        period = 1000.,
    )
    ```

    Here the rate will become 10Hz again every 1 second of simulation. If the period is smaller than the schedule, the remaining rates will not be set.

    Note that you can use the ``reset()`` method to manually reinitialize the schedule, times becoming relative to that call:

    ```python
    simulate(1200.) # Should switch to 100 Hz due to the period of 1000.
    inp.reset()
    simulate(1000.) # Starts at 10 Hz again.
    ```

    The rates were here global to the population. If you want each neuron to have a different rate, ``rates`` must have additional dimensions corresponding to the geometry of the population.

    ```python
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [ 
            [10. + 0.05*i for i in range(100)], 
            [20. + 0.05*i for i in range(100)],
        ],
        schedule = [0., 100.],
        period = 1000.,
    )
    ```

    """
    def __init__(self, geometry, rates, schedule, period= -1., name=None, copied=False):
        """    
        :param rates: array of firing rates. The first axis corresponds to the times where the firing rate should change. If a different rate should be used by the different neurons, the other dimensions must match the geometry of the population.
        :param schedule: list of times (in ms) where the firing rate should change.
        :param period: time when the timed array will be reset and start again, allowing cycling over the schedule. Default: no cycling (-1.).
        """
        
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

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied)

        # Check arguments
        try:
            rates = np.array(rates)
        except:
            Messages._error("TimedPoissonPopulation: the rates argument must be a numpy array.")

        schedule = np.array(schedule)

        nb_schedules = rates.shape[0]
        if nb_schedules != schedule.size:
            Messages._error("TimedPoissonPopulation: the first axis of the rates argument must be the same length as schedule.")


        if rates.ndim == 1 : # One rate for the whole population
            rates = np.array([np.full(self.size, rates[i]) for i in range(nb_schedules)]) 

        # Initial values
        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['period'] = period

    def _copy(self):
        "Returns a copy of the population when creating networks."
        return TimedPoissonPopulation(self.geometry, self.init['rates'] , self.init['schedule'], self.init['period'], self.name, copied=True)

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
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedPoissonPopulation
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

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
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': get_global_config('precision')}

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
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedPoissonPopulation
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

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
""" % {'float_prec': get_global_config('precision')}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': get_global_config('precision')}

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
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) {
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
    std::vector< std::vector< %(float_prec)s > > get_buffer() {
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
""" % {'float_prec': get_global_config('precision')}
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
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': get_global_config('precision')}
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_periodic(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id, 'float_prec': get_global_config('precision') }

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
    'float_prec': get_global_config('precision')
}

        self._specific_template['update_variable_header'] = "__global__ void cuPop%(id)s_local_step( const long int t, const double dt, curandState* rand_0, double* proba, unsigned int* num_events, int* spiked, long int* last_spike );" % {'id': self.id}
        # Please notice, that the GPU kernels can be launched only with one block. Otherwise, the
        # atomicAdd which is called inside the kernel is not working correct (HD: April 1st, 2021)
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s.update();

    // Reset old events
    clear_num_events<<< 1, 1, 0, pop%(id)s.stream >>>(pop%(id)s.gpu_spike_count);
#ifdef _DEBUG
    cudaError_t err_clear_num_events_%(id)s = cudaGetLastError();
    if(err_clear_num_events_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_clear_num_events_%(id)s) << std::endl;
#endif

    // Compute current events
    cuPop%(id)s_local_step<<< 1, pop%(id)s._threads_per_block, 0, pop%(id)s.stream >>>(
        t, dt,
        pop%(id)s.gpu_rand_0,
        pop%(id)s.gpu_proba,
        pop%(id)s.gpu_spike_count,
        pop%(id)s.gpu_spiked,
        pop%(id)s.gpu_last_spike
    );
#ifdef _DEBUG
    cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
    if(err_pop_spike_gather_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
#endif

    // transfer back the spike counter (needed by record)
    cudaMemcpyAsync( &pop%(id)s.spike_count, pop%(id)s.gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, pop%(id)s.stream );
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike_count: " << cudaGetErrorString(err) << std::endl;
#endif

    // transfer back the spiked array (needed by record)
    cudaMemcpyAsync( pop%(id)s.spiked.data(), pop%(id)s.gpu_spiked, pop%(id)s.spike_count*sizeof(int), cudaMemcpyDeviceToHost, pop%(id)s.stream );
#ifdef _DEBUG
    err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike: " << cudaGetErrorString(err) << std::endl;
#endif

""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = "//TODO: "

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                self.cyInstance.set_schedule( value / get_global_config('dt') )
            else:
                self.init['schedule'] = value
        elif name == 'rates':
            if self.initialized:
                value = np.array(value)
                if value.shape[0] != self.schedule.shape[0]:
                    Messages._error("TimedPoissonPopulation: the first dimension of rates must match the schedule.")
                if value.ndim > 2:
                    # we need to flatten the provided data
                    values = value.reshape( (value.shape[0], self.size) )
                    self.cyInstance.set_rates(values)
                elif value.ndim == 2:
                    if value.shape[1] != self.size:
                        if value.shape[1] == 1:
                            value = np.array([np.full(self.size, value[i]) for i in range(value.shape[0])])
                        else:
                            Messages._error("TimedPoissonPopulation: the second dimension of rates must match the number of neurons.")
                    self.cyInstance.set_rates(value)
                elif value.ndim == 1:
                    value = np.array([np.full(self.size, value[i]) for i in range(value.shape[0])]) 
                    self.cyInstance.set_rates(value)

            else:
                self.init['rates'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /get_global_config('dt')))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return get_global_config('dt') * self.cyInstance.get_schedule()
            else:
                return self.init['schedule']
        elif name == 'rates':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_rates()
                    values = np.zeros( tuple( [len(self.schedule)] + list(self.geometry) ) )
                    for x in range(len(self.schedule)):
                        values[x] = np.reshape( flat_values[x], self.geometry)
                    return values
                else:
                    return self.cyInstance.get_rates()
            else:
                return self.init['rates']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * get_global_config('dt')
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)