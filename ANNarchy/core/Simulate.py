"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.core import Global
from ANNarchy.core.Population import Population

from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern import Messages


from math import ceil
import time
import operator

# Callbacks
_callbacks = [[]]
_callbacks_enabled = [True]


def simulate(
        duration:float, 
        measure_time:bool=False, 
        progress_bar:bool=False, 
        callbacks:bool=True, 
        net_id:int=0) -> None:
    """
    Simulates the network for the given duration in milliseconds. 
    
    The number of simulation steps is computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms):

    ```python
    simulate(1000.0)
    ```

    :param duration: the duration in milliseconds.
    :param measure_time: defines whether the simulation time should be printed. 
    :param progress_bar: defines whether a progress bar should be printed. 
    :param callbacks: defines if the callback method (decorator ``every`` should be called).
    :returns:
    """
    if Profiler().enabled:
        t0 = time.time()

    if not NetworkManager().cy_instance(net_id=net_id):
        Messages._error('simulate(): the network is not compiled yet.')

    # Compute the number of steps
    nb_steps = ceil(float(duration) / Global.dt())

    if measure_time:
        tstart = time.time()

    if callbacks and _callbacks_enabled[net_id] and len(_callbacks[net_id]) > 0:
        _simulate_with_callbacks(duration, progress_bar, net_id)
    else:
        NetworkManager().cy_instance(net_id=net_id).pyx_run(nb_steps, progress_bar)

    if measure_time:
        if net_id > 0:
            Messages._print('Simulating', duration/1000.0, 'seconds of the network', net_id, 'took', time.time() - tstart, 'seconds.')
        else:
            Messages._print('Simulating', duration/1000.0, 'seconds of the network took', time.time() - tstart, 'seconds.')

    # Store the Python and C++ timings. Please note, that the C++ core
    # measures in ms and Python measures in s
    if Profiler().enabled:
        t1 = time.time()
        Profiler().add_entry( t0, t1, "simulate", "simulate")

        # network single step
        overall_avg, _ = Profiler()._cpp_profiler.get_timing("network", "step")
        Profiler().add_entry(overall_avg * nb_steps, 100.0, "overall", "cpp core")

        # single operations for populations
        for pop in NetworkManager().get_populations(net_id=net_id):
            for func in ["step", "rng", "delay", "spike"]:
                avg_time, _ = Profiler()._cpp_profiler.get_timing(pop.name, func)
                Profiler().add_entry( avg_time * nb_steps, (avg_time/overall_avg)*100.0, pop.name+"_"+func, "cpp core")

        # single operations for projections
        for proj in NetworkManager().get_projections(net_id=net_id):
            for func in ["psp", "step", "post_event"]:
                avg_time, _ = Profiler()._cpp_profiler.get_timing(proj.name, func)
                Profiler().add_entry( avg_time * nb_steps, (avg_time/overall_avg)*100.0, proj.name+"_"+func, "cpp core")

        monitor_avg, _ = Profiler()._cpp_profiler.get_timing("network", "record")
        Profiler().add_entry( monitor_avg * nb_steps, (monitor_avg/overall_avg)*100.0, "record", "cpp core")

def simulate_until(max_duration:float, population: Population | list[Population], operator='and', measure_time:bool = False, net_id:int=0):
    """
    Runs the network for the maximal duration in milliseconds. If the ``stop_condition`` defined in the population becomes true during the simulation, it is stopped.

    One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

    Example:

    ```python
    pop1 = Population( ..., stop_condition = "r > 1.0 : any")
    compile()
    simulate_until(max_duration=1000.0, population=pop1)
    ```

    :param max_duration: Maximum duration of the simulation in milliseconds.
    :param population: (list of) population(s) whose ``stop_condition`` should be checked to stop the simulation.
    :param operator: Operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
    :param measure_time: Defines whether the simulation time should be printed (default=False).
    :return: the actual duration of the simulation in milliseconds.
    """
    if not NetworkManager().cy_instance(net_id):
        Messages._error('simulate_until(): the network is not compiled yet.')

    nb_steps = ceil(float(max_duration) / Global.dt())
    if not isinstance(population, list):
        population = [population]

    if measure_time:
        tstart = time.time()

    nb = NetworkManager().cy_instance(net_id).pyx_run_until(nb_steps, [pop.id for pop in population], True if operator=='and' else False)

    sim_time = float(nb) / Global.dt()
    if measure_time:
        Messages._print('Simulating', nb/Global.dt()/1000.0, 'seconds of the network took', time.time() - tstart, 'seconds.')
    return sim_time


def step(net_id=0):
    """
    Performs a single simulation step (duration = `dt`).
    """
    if not NetworkManager().cy_instance(net_id):
        Messages._error('simulate_until(): the network is not compiled yet.')

    NetworkManager().cy_instance(net_id).pyx_step()


################################
## Decorators
################################

def callbacks_enabled(net_id=0):
    """
    Returns True if callbacks are enabled for the network.
    """
    return _callbacks_enabled[net_id]

def disable_callbacks(net_id=0):
    """
    Disables all callbacks for the network.
    """
    _callbacks_enabled[net_id] = False

def enable_callbacks(net_id=0):
    """
    Enables all declared callbacks for the network.
    """
    _callbacks_enabled[net_id] = True

def clear_all_callbacks(net_id=0):
    """
    Clears the list of declared callbacks for the network.

    Cannot be undone!
    """
    _callbacks[net_id].clear()


class every :
    """
    Decorator to declare a callback method that will be called periodically during the simulation.

    Example of setting increasing inputs to a population every 100 ms, with an offset of 90 ms (or -10 ms relative to the period):

    ```python
    @every(period=100., offset=-10.)
    def step_input(n):
        pop.I = float(n) / 100.

    simulate(10000.)
    ```

    ``step_input()`` will be called at times 90, 190, ..., 9990 ms during the call to ``simulate()``.

    The method must accept only ``n`` as parameter (an integer being 0 the first time the method is called, and incremented afterwards) and can not return anything.

    The times at which the method is called are relative to the time when ``simulate()`` is called (if ``t`` is already 150 before calling ``simulate()``, the first call will then be made at ``t=240`` with the previous example).

    If multiple callbacks are defined, they will be called in the order of their declaration if they occur at the same time.

    ``wait`` can be combined with ``offset``, so if ``period=100.``, ``offset=50.`` and ``wait=500.``, the first call will be made 550 ms after the call to ``simulate()`
    
    :param period: interval in ms between two calls to the function. If less than ``dt``, will be called every step.
    :param offset: by default, the first call to the method will be made at the start of the simulation. The offset delays the call within the period (default: 0.0). Can be negative, in which case it will be counted from the end of the period.
    :param wait: allows to wait for a certain amount of time (in ms) before starting to call the method.

    """

    def __init__(self, period:float, offset:float=0., wait:float=0.0, net_id:int=0) -> None:

        self.period = max(float(period), Global.dt())
        self.offset = min(float(offset), self.period)
        self.wait = max(float(wait), 0.0)
        _callbacks[net_id].append(self)

    def __call__(self, f):
        
        # If there are decorator arguments, __call__() is only called
        # once, as part of the decoration process! You can only give
        # it a single argument, which is the function object.
        self.func = f
        return f


def _simulate_with_callbacks(duration, progress_bar, net_id=0):
    """
    Replaces simulate() when call_backs are defined.
    """
    t_start = Global.get_current_step(net_id)
    length = int(duration/Global.dt())

    # Compute the times
    times = []
    for c in _callbacks[net_id]:
        period = int(c.period/Global.dt())
        offset = int(c.offset/Global.dt()) % period
        wait = int(c.wait/Global.dt())

        moments = range(t_start + wait + offset, t_start + length, period)
        n = 0
        for m in moments:
            times.append((m, c, n))
            n += 1

    # Sort the times to be sure they are in the right order.
    times = sorted(times, key=operator.itemgetter(0))

    for time, callback, n in times:
        # Advance the simulation to the desired time
        if time != Global.get_current_step(net_id):
            NetworkManager().cy_instance(net_id).pyx_run(time-Global.get_current_step(net_id), progress_bar)
        # Call the callback
        callback.func(n)

    # Go to the end of the duration
    if Global.get_current_step(net_id) < t_start + length:
        NetworkManager().cy_instance(net_id).pyx_run(t_start + length - Global.get_current_step(net_id), progress_bar)
