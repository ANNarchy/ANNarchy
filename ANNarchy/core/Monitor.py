"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Population import Population
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Dendrite import Dendrite
from ANNarchy.core import Global

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages

import numpy as np
import re
import sys
from copy import copy, deepcopy
from typing import Any

# objects/functions that should be available by "from ANNarchy import *"
__all__ = ["Monitor", "raster_plot", "histogram", "population_rate", "smoothed_rate", "mean_fr", "inter_spike_interval", "coefficient_of_variation"]

class Monitor :
    """
    Monitoring class allowing to record easily parameters or variables from Population, PopulationView, Dendrite or Projection objects.

    Example:

    ```python
    m = Monitor(pop, ['g_exc', 'v', 'spike'], period=10.0)
    ```

    It is also possible to record the sum of inputs to each neuron in a rate-coded population:

    ```python
    m = Monitor(pop, ['sum(exc)', 'r'])
    ```

    :param obj: object to monitor. Must be a `Population`, `PopulationView`, `Dendrite` or `Projection` object.
    :param variables: single variable name or list of variable names to record (default: []).
    :param period: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).
    :param period_offset: determine the moment in ms of recording within the period (default 0). Must be smaller than **period**.
    :param start: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.
    """

    def __init__(self, obj: Any, variables:list=[], period:float=None, period_offset:float=None, start:bool=True, net_id:int=0):


        # Object to record (Population, PopulationView, Dendrite)
        self.object = obj
        self.cyInstance = None
        self.net_id = net_id
        self.name = 'Monitor'

        # Check type of the object
        if not isinstance(self.object, (Population, PopulationView, Dendrite, Projection)):
            Messages._error('Monitor: the object must be a Population, PopulationView, Dendrite or Projection object')

        # Variables to record
        if not isinstance(variables, list):
            self._variables = [variables]
        else:
            self._variables = variables

        # Sanity check: we want only record variables
        for var in self._variables:
            if var == "w" and var in self.object.variables:
                continue

            if var in self.object.parameters:
                Messages._error('Parameters are not recordable')

            if not var in self.object.variables and not var in ['spike', 'axon_spike'] and not var.startswith('sum('):
                Messages._error('Monitor: the object does not have an attribute named', var)

        # Period
        if not period:
            self._period = get_global_config('dt')
        else:
            self._period = float(period)

        # Period Offset
        if not period_offset:
            self._period_offset = 0
        else:
            # check validity
            if period_offset >= period:
                Messages._error("Monitor(): value of period_offset must be smaller than period.")
            else:
                self._period_offset = period_offset

        # Warn users when recording projections
        if isinstance(self.object, Projection) and self._period == get_global_config('dt'):
            Messages._warning('Monitor(): it is a bad idea to record synaptic variables of a projection at each time step!')

        # Start
        self._start = start
        self._recorded_variables = {}
        self._last_recorded_variables = {}

        # Add the monitor to the global variable
        self.id = NetworkManager().number_monitors(net_id=self.net_id)

        NetworkManager().add_monitor(self.net_id, self)

        if NetworkManager().is_compiled(self.net_id): # Already compiled
            self._init_monitoring()

    # Extend the period attribute
    @property
    def period(self) -> float:
        "Period of recording in ms"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period * get_global_config('dt')
    @period.setter
    def period(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period = int(val/get_global_config('dt'))

    # Extend the period_offset attribute
    @property
    def period_offset(self) -> float:
        "Shift of moment of time of recording in ms within a period"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period_offset * get_global_config('dt')

    @period_offset.setter
    def period_offset(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period_offset = int(val/get_global_config('dt'))

    # Extend the variables attribute
    @property
    def variables(self) -> list:
        "Returns a copy of the current variable list."
        return copy(self._variables)

    @variables.setter
    def variables(self, val):
        Messages._error("Modifying of a Monitors variable list is not allowed")

    def size_in_bytes(self) -> int:
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
        """
        Adds a variable to the list of recorded attributes.
        """
        if not var in self._variables:
            self._variables.append(var)

        self._recorded_variables[var] = {
            'start': [Global.get_current_step(self.net_id)],
            'stop': [None],
        }

        self._last_recorded_variables[var] = {
            'start': [Global.get_current_step(self.net_id)],
            'stop': [None],
        }

    def reset(self) -> None:
        """
        Reset the monitor to its initial state.
        """
        for var in self._variables:
            # Flush the data
            data = self.get(var)
            del data
            # Reinitializes the timings
            self._add_variable(var)

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
            self.ranks = list(self.object.ranks)
        else:
            self.ranks = [-1]

        # Create the wrapper
        period = int(self._period/get_global_config('dt'))
        period_offset = int(self._period_offset/get_global_config('dt'))
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(NetworkManager().cy_instance(self.net_id), 'PopRecorder'+str(self.object.id)+'_wrapper')(self.ranks, period, period_offset, offset)

        for var in self._variables:
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
        period = int(self._period/get_global_config('dt'))
        period_offset = int(self._period_offset / get_global_config('dt'))
        offset = Global.get_current_step(self.net_id) % period

        # Create the wrapper
        self.cyInstance = getattr(NetworkManager().cy_instance(self.net_id), 'ProjRecorder'+str(proj_id)+'_wrapper')(self.idx, period, period_offset, offset)

        # Add the variables
        for var in self._variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def start(self, variables:list=None, period:float=None) -> None:
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
            self.cyInstance.period = int(self._period/get_global_config('dt'))
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
                        Messages._print('\t', var, 'is a parameter, its value is constant')

                Messages._warning('Monitor: ' + var + ' can not be recorded ('+obj_desc+')')


    def pause(self) -> None:
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
                    obj_desc = 'population ' + self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between ' + self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Messages._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')

            self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)


    def resume(self) -> None:
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
                Messages._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')

            self._recorded_variables[var]['start'].append(Global.get_current_step(self.net_id))
            self._recorded_variables[var]['stop'].append(None)

    def stop(self) -> None:
        """
        Stops the recording.

        Warning: This will delete the content of the C++ object and all data not previously retrieved is lost.
        """
        try:
            self._variables = []
            self._recorded_variables = {}
            self.cyInstance.clear()
            self.cyInstance = None

        except:
            obj_desc = ''
            if isinstance(self.object, (Population, PopulationView)):
                obj_desc = 'population '+self.object.name
            elif isinstance(self.object, Projection):
                obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
            else:
                obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
            Messages._warning('Monitor:' + obj_desc + 'cannot be stopped')


    def get(self, variables:str | list[str]=None, 
            keep:bool=False, reshape:bool=False, force_dict:bool=False) -> dict:
        """
        Returns the recorded variables as a Numpy array (first dimension is time, second is neuron index).

        If a single variable name is provided, the recorded values for this variable are directly returned.
        If a list is provided or the argument left empty, a dictionary with all recorded variables is returned.

        The `spike` variable of a population will be returned as a dictionary of lists, where the spike times (in steps) for each recorded neurons are returned.

        :param variables: (list of) variables. By default, a dictionary with all variables is returned.
        :param keep: defines if the content in memory for each variable should be kept (default: False).
        :param reshape: transforms the second axis of the array to match the population's geometry (default: False).
        :return: Recorded variables
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

            # Update stopping time
            self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)

            self._last_recorded_variables[var]['start'] = self._recorded_variables[var]['start']
            self._last_recorded_variables[var]['stop'] = self._recorded_variables[var]['stop']

            if not keep:
                self._recorded_variables[var]['start'] = [Global.get_current_step(self.net_id)]
                self._recorded_variables[var]['stop'] = [None]

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
        return np.array(data, dtype=object)

    def times(self, variables:list[str]=None) -> dict:
        """
        Returns the start and stop times (in ms) of the recorded variables.

        It should only be called after a call to ``get()``, so that it describes when the variables have been recorded.

        :param variables: (list of) variables. By default, the times for all variables is returned.
        :returns: dictionary of start and stop times.
        """
        t = {}
        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self._variables

        for var in variables:
            # check for spelling mistakes
            if not var in self._variables:
                Messages._warning("Variable '"+str(var)+"' is not monitored.")
                continue

            t[var] = deepcopy(self._last_recorded_variables[var])

        return t

    ###############################
    ### Spike visualisation stuff
    ###############################
    def raster_plot(self, spikes:dict=None) -> tuple:
        """
        Returns two numpy arrays representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

        Example:

        ```python
        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        t, n = m.raster_plot()
        plt.plot(t, n, '.')
        ```

        or:

        ```python
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        spikes = m.get('spike')
        t, n = m.raster_plot(spikes)
        plt.plot(t, n, '.')
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :returns: spike times and neuron indices as numpy arrays..
        """
        times = []; ranks=[]
        if not 'spike' in self._variables:
            Messages._error('Monitor: spike was not recorded')

        # Get data
        if spikes is None:
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

    def histogram(self, spikes=None, bins=None, per_neuron=False, recording_window=None):
        """
        Returns a histogram for the recorded spikes in the population.

        Example:

        ```python
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        histo = m.histogram()
        plt.plot(histo)
        ```

        or:

        ```python
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        spikes = m.get('spike')
        histo = m.histogram(spikes)
        plt.plot(histo)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :param bins: the bin size in ms (default: dt).
        """
        if not 'spike' in self._variables:
            Messages._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        return histogram(data, bins=bins, per_neuron=per_neuron, recording_window=recording_window)

    def inter_spike_interval(self, spikes=None, ranks=None, per_neuron=False):
        """
        Computes the inter-spike interval for the recorded spikes in the population.

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :ranks:        a list of neurons that should be evaluated. By default (None), all neurons are evaluated.
        :per_neuron:   if set to True, the computed inter-spike intervals are stored per neuron (analog to spikes), otherwise all values are stored in one huge vector (default: False).
        """
        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        return inter_spike_interval(data, ranks=ranks, per_neuron=per_neuron)

    def coefficient_of_variation(self, spikes=None, ranks=None):
        """
        Computes the coefficient of variation for the recorded spikes in the population.

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :ranks:        a list of neurons that should be evaluated. By default (None), all neurons are evaluated.
        """
        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        return coefficient_of_variation(data, ranks=ranks)

    def mean_fr(self, spikes=None):
        """
        Computes the mean firing rate in the population during the recordings.

        Example:

        ```python
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        fr = m.mean_fr()
        ```

        or:

        ```python
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        spikes = m.get('spike')
        fr = m.mean_fr(spikes)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        """
        if not 'spike' in self._variables:
            Messages._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes


        # Compute the duration of the recordings
        duration = self._last_recorded_variables['spike']['stop'][-1] - self._last_recorded_variables['spike']['start'][-1]

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
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        r = m.smoothed_rate(smooth=100.)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        :param smooth: smoothing time constant. Default: 0.0 (no smoothing).

        """
        if not 'spike' in self._variables:
            Messages._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.cython_ext.Transformations as Transformations
        return Transformations.smoothed_rate(
            {
                'data': data,
                'start': self._last_recorded_variables['spike']['start'][-1],
                'stop': self._last_recorded_variables['spike']['stop'][-1]
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
        m = ann.Monitor(P[:1000], 'spike')
        ann.simulate(1000.0)
        r = m.population_rate(smooth=100.)
        ```

        :param spikes: the dictionary of spikes returned by ``get('spike')``.
        :param smooth: smoothing time constant. Default: 0.0 (no smoothing).

        """
        if not 'spike' in self._variables:
            Messages._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.cython_ext.Transformations as Transformations
        return Transformations.population_rate(
            {
                'data': data,
                'start': self._last_recorded_variables['spike']['start'][-1],
                'stop': self._last_recorded_variables['spike']['stop'][-1]
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

class MemoryStats :
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
        for pop in NetworkManager().get_populations(net_id=net_id):
            if hasattr(pop, 'size_in_bytes'):
                print(pop.name, ":", self._human_readable_bytes(pop.size_in_bytes()))
            else:
                Messages._warning("MemoryStats.print_cpp(): the object", pop, "does not have a size_in_bytes() function.")

        for proj in NetworkManager().get_projections(net_id=net_id):
            if hasattr(proj, 'size_in_bytes'):
                print(proj.pre.name, "->", proj.post.name, "(", proj.target, "):", self._human_readable_bytes(proj.size_in_bytes()))
            else:
                Messages._warning("MemoryStats.print_cpp(): the object", proj, "does not have a size_in_bytes() function.")

        for mon in NetworkManager().get_monitors(net_id=net_id):
            if hasattr(proj, 'size_in_bytes'):
                print("Monitor on", mon.object.name, ":", self._human_readable_bytes(mon.size_in_bytes()))
            else:
                Messages._warning("MemoryStats.print_cpp(): the object", mon, "does not have a size_in_bytes() function.")

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
def raster_plot(spikes:dict) -> tuple:
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

    Example:

    ```python
    m = ann.Monitor(P[:1000], 'spike')
    ann.simulate(1000.0)
    spikes = m.get('spike')
    t, n = raster_plot(spikes)

    plt.plot(t, n, '.')
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


def histogram(spikes:dict, bins:float=None, per_neuron:bool=False, recording_window:tuple=None):
    """
    Returns a histogram for the recorded spikes in the population.

    Example:

    ```python
    m = ann.Monitor(P[:1000], 'spike')
    ann.simulate(1000.0)
    spikes = m.get('spike')
    histo = histogram(spikes)
    plt.plot(histo)
    ```

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param bins: the bin size in ms (default: dt).
    """
    if bins is None:
        bins =  get_global_config('dt')

    bin_step = int(bins/get_global_config('dt'))

    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    if recording_window is None:
        t_max = np.max(t_maxes)
        t_min = np.min(t_mines)
    else:
        t_min = recording_window[0]
        t_max = recording_window[1]
    duration = t_max - t_min

    # Number of bins
    nb_bins = int(duration/bin_step)
    #print(t_min, t_max, duration, nb_bins)

    if per_neuron:
        max_rank = np.amax([x for x in spikes.keys()])+1
        # Initialize histogram
        histo = [ [0 for _ in range(nb_bins+1)] for _ in range(max_rank) ]

        # Compute per step histogram
        for neuron in spikes.keys():
            for t in spikes[neuron]:
                histo[neuron][int((t-t_min)/float(bin_step))] += 1

    else:
        # Initialize histogram
        histo = [0 for t in range(nb_bins+1)]

        # Compute per step histogram
        for neuron in spikes.keys():
            for t in spikes[neuron]:
                histo[int((t-t_min)/float(bin_step))] += 1

    return np.array(histo)

def inter_spike_interval(spikes:dict, ranks:list=None, per_neuron:bool=False):
    """
    Computes the inter-spike interval (ISI) for the recorded spike events of a population.

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param ranks: list of ranks.
    :param per_neuron: if True, the ISI will be computed per neuron, not globally.

    """
    isi = {}
    for neuron_rank, spike_events in spikes.items():
        # ISI computation requires at least 2 events
        if len(spike_events) < 2:
            continue

        # suppress unwanted neurons
        if ranks is not None:
            if neuron_rank not in ranks:
                continue

        # compute time difference between spike events
        tmp_isi=[]
        for idx in range(len(spike_events)-1):
            tmp_isi.append((spike_events[idx+1]-spike_events[idx])*Global.dt())

        isi[neuron_rank] = tmp_isi

    if per_neuron:
        return isi
    else:
        res = []
        for val in isi.values():
            res.extend(val)
        return res

def coefficient_of_variation(spikes:dict, ranks:list=None, per_neuron:bool=False):
    """
    Computes the coefficient of variation of the inter-spike intervals for the recorded spike events of a population.

    :param spikes: the dictionary of spikes returned by ``get('spike')``.
    :param ranks: list of ranks.
    :param per_neuron: if True, the ISI will be computed per neuron, not globally.
    """
    isi_per_neuron = inter_spike_interval(spikes, ranks=ranks, per_neuron=True)
    isi_cv = {}
    for neuron_rank, values in isi_per_neuron.items():
        if len(values) < 2:
            continue     # no meaningful mean/std possible

        # suppress unwanted neurons
        if ranks is not None:
            if neuron_rank not in ranks:
                continue

        isi_cv[neuron_rank] = np.std(values) / np.mean(values)

    if per_neuron:
        return isi_cv
    else:
        res = []
        for val in isi_cv.values():
            res.append(val)
        return res

def population_rate(spikes:dict, smooth:float=0.0):
    """
    Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

    This method is faster than calling ``smoothed_rate`` and then averaging.

    The first axis is the neuron index, the second is time.

    Example:

    ```python
    m = ann.Monitor(P[:1000], 'spike')
    ann.simulate(1000.0)
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

    import ANNarchy.cython_ext.Transformations as Transformations
    return Transformations.population_rate(
        {
            'data': spikes,
            'start':t_min,
            'stop': t_max
        },
        smooth
    )

def smoothed_rate(spikes:dict, smooth:float=0.):
    """
    Computes the smoothed firing rate of the recorded spiking neurons.

    The first axis is the neuron index, the second is time.

    Example:

    ```python
    m = ann.Monitor(P[:1000], 'spike')
    ann.simulate(1000.0)
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

    import ANNarchy.cython_ext.Transformations as Transformations
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
    m = ann.Monitor(P[:1000], 'spike')
    ann.simulate(1000.0)
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
