#===============================================================================
#
#     Monitor.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from . import Global
from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Dendrite import Dendrite

import numpy as np
import re
import sys

class Monitor(object):
    """
    Monitoring class allowing to record easily parameters or variables from Population, PopulationView and Dendrite objects.
    
    Example:

    ```python
    m = Monitor(pop, ['g_exc', 'v', 'spike'], period=10.0)
    ```

    It is also possible to record the sum of inputs to each neuron in a rate-coded population:

    ```python
    m = Monitor(pop, ['sum(exc)', 'r'])
    ```

    """

    def __init__(self, obj, variables=[], period=None, period_offset=None, start=True, net_id=0):
        """
        :param obj: object to monitor. Must be a Population, PopulationView, Dendrite or Projection object.
        :param variables: single variable name or list of variable names to record (default: []).
        :param period: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).
        :param period_offset: determine the moment in ms of recording within the period (default 0). Must be smaller than **period**.
        :param start: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.
        """
        # Object to record (Population, PopulationView, Dendrite)
        self.object = obj
        self.cyInstance = None
        self.net_id = net_id
        self.name = 'Monitor'

        # Check type of the object
        if not isinstance(self.object, (Population, PopulationView, Dendrite, Projection)):
            Global._error('Monitor: the object must be a Population, PopulationView, Dendrite or Projection object')

        # Variables to record
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        # Check variables
        for var in self.variables:
            if not var in self.object.attributes and not var in ['spike', 'axon_spike'] and not var.startswith('sum('):
                Global._error('Monitor: the object does not have an attribute named', var)

        # Period
        if not period:
            self._period = Global.config['dt']
        else:
            self._period = float(period)

        # Period Offset
        if not period_offset:
            self._period_offset = 0
        else:
            # check validity
            if period_offset >= period:
                Global._error("Monitor(): value of period_offset must be smaller than period.")
            else:
                self._period_offset = period_offset

        # Warn users when recording projections
        if isinstance(self.object, Projection) and self._period == Global.config['dt']:
            Global._warning('Monitor(): it is a bad idea to record synaptic variables of a projection at each time step!')

        # Start
        self._start = start
        self._recorded_variables = {}

        # Add the monitor to the global variable
        self.id = len(Global._network[self.net_id]['monitors'])

        Global._network[self.net_id]['monitors'].append(self)

        if Global._network[self.net_id]['compiled']: # Already compiled
            self._init_monitoring()

    # Extend the period attribute
    @property
    def period(self):
        "Period of recording in ms"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period * Global.config['dt']
    @period.setter
    def period(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period = int(val/Global.config['dt'])

    # Extend the period_offset attribute
    @property
    def period_offset(self):
        "Shift of moment of time of recording in ms within a period"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period_offset * Global.config['dt']

    @period_offset.setter
    def period_offset(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period_offset = int(val/Global.config['dt'])

    def size_in_bytes(self):
        """
        Get the size of allocated memory on C++ side. Please note, this is only valid if compile() was invoked.

        :return: size in bytes of all allocated C++ data.
        """
        if hasattr(self.cyInstance, 'size_in_bytes'):
            return self.cyInstance.size_in_bytes()

    def _clear(self):
        """
        Deallocates the container within the C++ instance. The population object is not usable anymore after calling this function.

        Warning: should be only called by the net deconstructor (in the context of parallel_run).
        """
        if hasattr(self.cyInstance, 'clear'):
            self.cyInstance.clear()


    def _add_variable(self, var):
        if not var in self.variables:
            self.variables.append(var)
        self._recorded_variables[var] = {'start': [Global.get_current_step(self.net_id)], 'stop': [Global.get_current_step(self.net_id)]}

    def _init_monitoring(self):
        "To be called after compile() as it accesses cython objects"
        # Start recording dependent on the recorded object
        from ANNarchy.extensions.bold import BoldMonitor
        if isinstance(self, BoldMonitor):
            self._start_bold_monitor() # pylint: disable=no-member
        elif isinstance(self.object, (Population, PopulationView)):
            self._start_population()
        elif isinstance(self.object, (Dendrite, Projection)):
            self._start_dendrite()

    def _start_population(self):
        "Creates the C++ object and starts the recording for a population."

        if isinstance(self.object, PopulationView):
            self.ranks = self.object.ranks
        else:
            self.ranks = [-1]

        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'PopRecorder'+str(self.object.id)+'_wrapper')(self.ranks, period, period_offset, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        for var in self.variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def _start_dendrite(self):
        "Creates the C++ object and starts the recording for a dendrite."

        if isinstance(self.object, Dendrite):
            self.ranks = self.object.post_rank
            self.idx = [self.object.idx]
            proj_id = self.object.proj.id
        else: # Projection
            self.ranks = [-1]
            self.idx = self.object.post_ranks
            proj_id = self.object.id

        # Compute the period and offset
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset / Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period

        # Create the wrapper
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'ProjRecorder'+str(proj_id)+'_wrapper')(self.idx, period, period_offset, offset)

        # Add the monitor to the network
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        # Add the variables
        for var in self.variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def _clear(self):
        """
        Clear the C++ data if a _clear method was defined.
        """
        if Global._network[self.net_id]['instance']:
            Global._network[self.net_id]['instance'].remove_recorder(self.cyInstance)
        if hasattr(self.cyInstance, "clear"):
            self.cyInstance.clear()

    def start(self, variables=None, period=None):
        """Starts recording the variables. 
        
        It is called automatically after ``compile()`` if the flag ``start`` was not passed to the constructor.

        :param variables: single variable name or list of variable names to start recording (default: the ``variables`` argument passed to the constructor).
        :param period: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).
        """
        if variables:
            if not isinstance(variables, list):
                self._add_variable(variables)
                variables = [variables]
            else:
                for var in variables:
                    self._add_variable(var)
        else:
            variables = self.variables

        if period:
            self._period = period
            self.cyInstance.period = int(self._period/Global.config['dt'])
            self.cyInstance.offset = Global.get_current_step(self.net_id)

        for var in variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, True)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population ' + self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                    if var in self.object.proj.parameters:
                        Global._print('\t', var, 'is a parameter, its value is constant')
                Global._warning('Monitor: ' + var + ' can not be recorded ('+obj_desc+')')

    def resume(self):
        "Resumes the recordings."
        # Start recording the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, True)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')
            self._recorded_variables[var]['start'].append(Global.get_current_step(self.net_id))

    def pause(self):
        "Pauses the recordings."
        # Start recording the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, False)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')
            self._recorded_variables[var]['stop'].append(Global.get_current_step(self.net_id))

    def stop(self):
        "Stops the recordings."
        # Stop and clear the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, False)
                getattr(self.cyInstance, 'clear_'+name)()
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')

        self.variables = []
        self._recorded_variables = {}
        Global._network[0]['instance'].remove_recorder(self.cyInstance)
        self.cyInstance = None


    def get(self, variables=None, keep=False, reshape=False, force_dict=False):
        """
        Returns the recorded variables as a Numpy array (first dimension is time, second is neuron index).

        If a single variable name is provided, the recorded values for this variable are directly returned.
        If a list is provided or the argument left empty, a dictionary with all recorded variables is returned.

        The ``spike`` variable of a population will be returned as a dictionary of lists, where the spike times (in steps) for each recorded neurons are returned.

        :param variables: (list of) variables. By default, a dictionary with all variables is returned.
        :param keep: defines if the content in memory for each variable should be kept (default: False).
        :param reshape: transforms the second axis of the array to match the population's geometry (default: False).
        """

        def reshape_recording(self, data):
            if not reshape:
                return data
            else:
                return data.reshape((data.shape[0],) + self.object.geometry)

        def return_variable(self, name, keep):
            if isinstance(self.object, (Population, PopulationView)):
                return reshape_recording(self, self._get_population(self.object, name, keep))
            elif isinstance(self.object, (Dendrite, Projection)):
                data = self._get_dendrite(self.object, name, keep)
                # Dendrites have one empty dimension
                if isinstance(self.object, Dendrite):
                    data = data.squeeze()
                return data
            else:
                return None


        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self.variables
            force_dict = True

        data = {}
        for var in variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target

            # Retrieve the data
            data[var] = return_variable(self, name, keep)

            # Eventually reshape the array
            try:
                if not keep:
                    if self._recorded_variables[var]['stop'][-1] != Global.get_current_step(self.net_id):
                        self._recorded_variables[var]['start'][-1] = self._recorded_variables[var]['stop'][-1]
                        self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)
                else:
                    if self._recorded_variables[var]['stop'][-1] != Global.get_current_step(self.net_id):
                        self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)
            except:
                Global._warning('Monitor.get(): you try to get recordings which do not exist:', var)

        if not force_dict and len(variables)==1:
            return data[variables[0]]
        else:
            return data

    def _get_population(self, pop, name, keep):
        try:
            data = getattr(self.cyInstance, name)
            if not keep:
                getattr(self.cyInstance, 'clear_' + name)()
        except:
            data = []

        if name not in ['spike', 'axon_spike']:
            return np.array(data)
        else:
            return data

    def _get_dendrite(self, proj, name, keep):
        try:
            data = getattr(self.cyInstance, name)
            if not keep:
                getattr(self.cyInstance, 'clear_' + name)()
        except:
            data = []
        return np.array(data)

    def times(self, variables=None):
        """ Returns the start and stop times of the recorded variables.

        :param variables: (list of) variables. By default, the times for all variables is returned.
        """
        import copy
        t = {}
        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self.variables
        for var in variables:
            if not var in self.variables:
                continue
            t[var] = copy.deepcopy(self._recorded_variables[var])
        return t

    ###############################
    ### Spike visualisation stuff
    ###############################
    def raster_plot(self, spikes=None):
        """
        Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spike_times, spike_ranks = m.raster_plot()
        plt.plot(spike_times, spike_ranks, '.')
        ```

        or:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        spike_times, spike_ranks = m.raster_plot(spikes)
        plt.plot(spike_times, spike_ranks, '.')
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        """
        times = []; ranks=[]
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            elif 'axon_spike' in spikes.keys():
                data = spikes['axon_spike']
            else:
                data = spikes

        # Compute raster
        for n in data.keys():
            for t in data[n]:
                times.append(t)
                ranks.append(n)

        return Global.dt()* np.array(times), np.array(ranks)

    def histogram(self, spikes=None, bins=None):
        """
        Returns a histogram for the recorded spikes in the population.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        histo = m.histogram()
        plt.plot(histo)
        ```

        or:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        histo = m.histogram(spikes)
        plt.plot(histo)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :param bins: the bin size in ms (default: dt).

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        if not bins:
            bins =  Global.config['dt']

        # Compute the duration of the recordings
        t_start = self._recorded_variables['spike']['start'][-1]
        duration = self._recorded_variables['spike']['stop'][-1] - self._recorded_variables['spike']['start'][-1]

        # Number of bins
        nb_bins = int(duration*Global.config['dt']/bins)

        # Initialize histogram
        histo = [0 for t in range(nb_bins)]

        # Compute histogram
        neurons = self.object.ranks if isinstance(self.object, PopulationView) else range(self.object.size)
        for neuron in neurons:
            for t in data[neuron]:
                histo[int((t-t_start)/float(bins/Global.config['dt']))] += 1

        return np.array(histo)

    def mean_fr(self, spikes=None):
        """
        Computes the mean firing rate in the population during the recordings.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        fr = m.mean_fr()
        ```

        or:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        fr = m.mean_fr(spikes)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes


        # Compute the duration of the recordings
        duration = self._recorded_variables['spike']['stop'][-1] - self._recorded_variables['spike']['start'][-1]

        # Number of neurons
        neurons = self.object.ranks if isinstance(self.object, PopulationView) else range(self.object.size)

        # Compute fr
        fr = 0
        for neuron in neurons:
            fr += len(data[neuron])

        return fr/float(len(neurons))/duration/Global.dt()*1000.0



    def smoothed_rate(self, spikes=None, smooth=0.):
        """
        Computes the smoothed firing rate of the recorded spiking neurons.

        The first axis is the neuron index, the second is time.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        r = m.smoothed_rate(smooth=100.)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :param smooth: smoothing time constant. Default: 0.0 (no smoothing).

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.core.cython_ext.Transformations as Transformations
        return Transformations.smoothed_rate(
            {
                'data': data,
                'start': self._recorded_variables['spike']['start'][-1],
                'stop': self._recorded_variables['spike']['stop'][-1]
            },
            smooth
        )

    def population_rate(self, spikes=None, smooth=0.):
        """
        Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

        This method is faster than calling ``smoothed_rate`` and then averaging.

        The first axis is the neuron index, the second is time.

        If ``spikes`` is left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        r = m.population_rate(smooth=100.)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``.
        :param smooth: smoothing time constant. Default: 0.0 (no smoothing).

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.core.cython_ext.Transformations as Transformations
        return Transformations.population_rate(
            {
                'data': data,
                'start': self._recorded_variables['spike']['start'][-1],
                'stop': self._recorded_variables['spike']['stop'][-1]
            },
            smooth
        )

def get_size(obj, seen=None):
    """
    Recursively determines the size of objects
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

class MemoryStats(object):
    """
    Create memory statistics for the main objects in ANNarchy. The current implementation
    focusses on the C++ simulation core. But this module could be further extended to measure
    also the Python objects.
    """
    def __init__(self):
        pass

    def print_cpp(self, net_id=0):
        """
        Print memory consumption of CPP objects. The method calls
        the size_in_bytes() methods implemented by the C++ modules.
        """
        for pop in Global._network[net_id]['populations']:
            if hasattr(pop, 'size_in_bytes'):
                print(pop.name, ":", self._human_readable_bytes(pop.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", pop, "does not have a size_in_bytes() function.")

        for proj in Global._network[net_id]['projections']:
            if hasattr(proj, 'size_in_bytes'):
                print(proj.pre.name, "->", proj.post.name, "(", proj.target, "):", self._human_readable_bytes(proj.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", proj, "does not have a size_in_bytes() function.")

        for mon in Global._network[net_id]['monitors']:
            if hasattr(proj, 'size_in_bytes'):
                print("Monitor on", mon.object.name, ":", self._human_readable_bytes(mon.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", mon, "does not have a size_in_bytes() function.")

    def _human_readable_bytes(self, num):
        """
        All cpp functions return there size in bytes *num* as long int. This function
        divides this by 1024 until the result is lower than the next unit.
        """
        for x in ['bytes','KB','MB','GB']:
            if num < 1024.0:
                return "%3.2f %s" % (num, x)
            num /= 1024.0
        return "%3.1f%s" % (num, 'TB')

######################
# Static methods to plot spike patterns without a Monitor (e.g. offline)
######################
def raster_plot(spikes):
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

    Example:

    ```python
    m = Monitor(P[:1000], 'spike')
    simulate(1000.0)
    spikes = m.get('spike')
    spike_times, spike_ranks = raster_plot(spikes)
    plt.plot(spike_times, spike_ranks, '.')
    ```

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    """
    times = []; ranks=[]

    # Compute raster
    for n in spikes.keys():
        for t in spikes[n]:
            times.append(t)
            ranks.append(n)

    return Global.dt()* np.array(times), np.array(ranks)


def histogram(spikes, bins=None):
    """
    Returns a histogram for the recorded spikes in the population.

    Example:

    ```python
    m = Monitor(P[:1000], 'spike')
    simulate(1000.0)
    spikes = m.get('spike')
    histo = histogram(spikes)
    plt.plot(histo)
    ```


    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param bins: the bin size in ms (default: dt).
    """
    if bins is None:
        bins =  Global.config['dt']

    bin_step = int(bins/Global.config['dt'])

    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)
    duration = t_max - t_min

    # Number of bins
    nb_bins = int(duration/bin_step)
    print(t_min, t_max, duration, nb_bins)

    # Initialize histogram
    histo = [0 for t in range(nb_bins+1)]

    # Compute per step histogram
    for neuron in spikes.keys():
        for t in spikes[neuron]:
            histo[int((t-t_min)/float(bin_step))] += 1

    return np.array(histo)

def population_rate(spikes, smooth=0.0):
    """
    Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

    This method is faster than calling ``smoothed_rate`` and then averaging.

    The first axis is the neuron index, the second is time.

    Example:

    ```python
    m = Monitor(P[:1000], 'spike')
    simulate(1000.0)
    spikes = m.get('spike')
    r = population_rate(smooth=100.)
    ```

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param smooth: smoothing time constant. Default: 0.0 (no smoothing).
    """
    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)

    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.population_rate(
        {
            'data': spikes,
            'start':t_min,
            'stop': t_max
        },
        smooth
    )

def smoothed_rate(spikes, smooth=0.):
    """
    Computes the smoothed firing rate of the recorded spiking neurons.

    The first axis is the neuron index, the second is time.

    Example:

    ```python
    m = Monitor(P[:1000], 'spike')
    simulate(1000.0)
    spikes = m.get('spike')
    r = smoothed_rate(smooth=100.)
    ```


    :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
    :param smooth: smoothing time constant. Default: 0.0 (no smoothing).
    """
    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)

    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.smoothed_rate(
        {
            'data': spikes,
            'start': t_min,
            'stop': t_max
        },
        smooth
    )

def mean_fr(spikes, duration=None):
    """
    Computes the mean firing rate in the population during the recordings.

    Example:

    ```python
    m = Monitor(P[:1000], 'spike')
    simulate(1000.0)
    spikes = m.get('spike')
    fr = mean_fr(spikes)
    ```

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param duration: duration of the recordings. By default, the mean firing rate is computed between the first and last spikes of the recordings.


    """
    if duration is None:

        # Compute the duration of the recordings
        t_maxes = []
        t_mines = []
        for neuron in spikes.keys():
            if len(spikes[neuron]) == 0 : continue
            t_maxes.append(np.max(spikes[neuron]))
            t_mines.append(np.min(spikes[neuron]))

        t_max = np.max(t_maxes)
        t_min = np.min(t_mines)
        duration = t_max - t_min

    nb_neurons = len(spikes.keys())

    # Compute fr
    fr = 0
    for neuron in spikes:
        fr += len(spikes[neuron])

    return fr/float(nb_neurons)/duration/Global.dt()*1000.0
