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

class TimedArray(SpecificPopulation):
    """
    Data structure holding sequential inputs for a rate-coded network.

    The input values are stored in the (recordable) attribute `r`, without any further processing.

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

    net = ann.Network()
    inp = net.create(ann.TimedArray(rates=inputs))
    pop = net.create(10, ann.LeakyIntegrator)

    proj = net.connect(inp, pop, 'exc')
    proj.one_to_one(1.0)

    net.compile()

    net.simulate(10.)
    ```

    This creates a population of 10 neurons whose activity will change during the first 10*dt milliseconds of the simulation. After that delay, the last input will be kept (i.e. 1 for the last neuron).

    If you want the TimedArray to "loop" over the different input vectors, you can specify a period for the inputs:

    ```python
    inp = net.create(ann.TimedArray(rates=inputs, period=10.))
    ```

    If the period is smaller than the length of the rates, the last inputs will not be set.

    If you do not want the inputs to be set at every step, but every 10 ms for example, you can use the ``schedule`` argument:

    ```python
    inp = net.create(ann.TimedArray(rates=inputs, schedule=10.))
    ```

    The input [1, 0, 0,...] will stay for 10 ms, then [0, 1, 0, ...] for the next 10 ms, etc...

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
                 copied:bool=False,
                 net_id:int=0,
                 ):

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

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied, net_id=net_id)

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

    def update(self, rates:np.ndarray, schedule:float=0., period:float=-1) -> None:
        """
        Set a new array of inputs. 
        
        The first axis corresponds to time, the others to the desired dimensions of the population. Note, the
        geometry is set during construction phase of the object.

        :param rates: array of firing rates. The first axis corresponds to time, the others to the desired dimensions of the population.
        :param schedule: either a single value or a list of time points where inputs should be set. Default: every timestep.
        :param period: time when the timed array will be reset and start again, allowing cycling over the inputs. Default: no cycling (-1.).
        """

        # Check the schedule
        if isinstance(schedule, (int, float)):
            if float(schedule) <= 0.0:
                schedule = ConfigManager().get('dt', self.net_id)

            self.schedule = [ float(schedule*i) for i in range(rates.shape[0])]
        else:
            self.schedule = schedule

        if len(self.schedule) > rates.shape[0]:
            Messages._error('TimedArray: the length of the schedule parameter cannot exceed the first dimension of the rates parameter.')

        if len(self.schedule) < rates.shape[0]:
            Messages._warning('TimedArray: the length of the schedule parameter is smaller than the first dimension of the rates parameter (more data than time points). Make sure it is what you expect.')

        self.rates = np.array(rates)
        self.period = period

    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return TimedArray(
            rates=self.rates, 
            geometry=self.geometry, 
            schedule=self.schedule, 
            period=self.period, 
            name=self.name, 
            copied=True, 
            net_id=self.net_id if net_id is None else net_id)

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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedArray
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[_block]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        # Nanobind
        self._specific_template['wrapper'] = f"""
    // TimedArray
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_buffer)
        .def("get_rates", &PopStruct{self.id}::get_buffer)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);

"""

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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedArray
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[_block]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        

        # Nanobind
        self._specific_template['wrapper'] = f"""
    // TimedArray
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_buffer)
        .def("get_rates", &PopStruct{self.id}::get_buffer)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);

"""

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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) {
    #ifdef _DEBUG
        std::cout << "PopStruct%(id)s::set_buffer()" << std::endl;
    #endif
        // clear a previous allocated container.
        if ( !_gpu_buffer.empty() ) {
            for (auto it = _gpu_buffer.begin(); it != _gpu_buffer.begin(); it++) {
                cudaFree(*it);
            }
            _gpu_buffer.clear();
            _gpu_buffer.shrink_to_fit();
        }
        // abort updating the container if no data is provided.
        if (buffer.empty()) {
            std::cerr << "The buffer provided to TimedArray should not be empty!" << std::endl;
            gpu_r = nullptr;
            _gpu_buffer = _gpu_buffer = std::vector< %(float_prec)s* >();
            return;
        }

        // host holds a set of pointers
        _gpu_buffer = std::vector< %(float_prec)s* >(buffer.size(), nullptr);

        // allocate gpu arrays
        for(int i = 0; i < buffer.size(); i++) {
            cudaMalloc((void**)&_gpu_buffer[i], buffer[i].size()*sizeof(%(float_prec)s));
        }

        auto host_it = buffer.begin();
        auto dev_it = _gpu_buffer.begin();
        for (; host_it != buffer.end(); host_it++, dev_it++) {
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
""" % {'id': self.id, 'float_prec': ConfigManager().get('precision', self.net_id)}
        
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
        if (!_gpu_buffer.empty())
            gpu_r = _gpu_buffer[0];
"""

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
    pop%(id)s->update();
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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        

        # Nanobind
        self._specific_template['wrapper'] = f"""
    // TimedArray
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)
        .def_rw("r_host_to_device", &PopStruct{self.id}::r_host_to_device)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_rates", &PopStruct{self.id}::set_buffer)
        .def("get_rates", &PopStruct{self.id}::get_buffer)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods

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
                val_int = np.array((np.atleast_1d(value) / ConfigManager().get('dt', self.net_id)), dtype=np.int32)
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
                self.cyInstance.set_period(int(value /ConfigManager().get('dt', self.net_id)))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return [ConfigManager().get('dt', self.net_id) * val for val in self.cyInstance.get_schedule()]
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
                return self.cyInstance.get_period() * ConfigManager().get('dt', self.net_id)
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)
        

