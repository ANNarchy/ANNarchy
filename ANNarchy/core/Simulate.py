"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.core.Population import Population
from ANNarchy.core import Global

from ANNarchy.intern.Profiler import Profiler
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

from math import ceil
import time
import tqdm
import operator

__all__ = [
    "simulate",
    "simulate_until",
    "step",
    "callbacks_enabled",
    "disable_callbacks",
    "enable_callbacks",
    "clear_all_callbacks",
    "every"
]

def simulate(
        duration:float, 
        measure_time:bool=False, 
        callbacks:bool=True, 
        net_id:int=0) -> None:
    """
    Simulates the network for the given duration in milliseconds. 
    
    The number of simulation steps is computed relative to the discretization step ``dt`` declared in ``setup()`` (default: 1ms):

    ```python
    ann.simulate(1000.0)
    ```

    :param duration: the duration in milliseconds.
    :param measure_time: defines whether the simulation time should be printed. 
    :param callbacks: defines if the callback methods (decorator ``every``) should be called.
    """
    if Profiler().enabled:
        t0 = time.time()
    
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)

    # Sanity checks
    if not network.compiled:
        Messages._error('simulate(): the network is not compiled yet.')
    if not network.instance:
        Messages._error('simulate(): the network is not initialized yet.')

    # Compute the number of steps
    nb_steps = ceil(float(duration) / ConfigManager().get("dt", net_id))

    if measure_time:
        tstart = time.time()

    if callbacks and network._callbacks_enabled and len(network._callbacks) > 0:
        _simulate_with_callbacks(duration, net_id)
    else:
        batch = 1000 # someday find a better solution...
        if nb_steps < batch:
            network.instance.run(nb_steps)
        else:
            nb = int(nb_steps/batch)
            rest = nb_steps % batch

            for i in range(nb):
                network.instance.run(batch)

            if rest > 0:
                network.instance.run(rest)

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
        overall_avg = Profiler()._cpp_profiler.get_avg_time("network", "step")
        Profiler().add_entry(overall_avg * nb_steps, 100.0, "overall", "cpp core")

        # single operations for populations
        for pop in network.get_populations():
            for func in ["step", "rng", "delay", "spike"]:
                avg_time = Profiler()._cpp_profiler.get_avg_time(pop.name, func)
                Profiler().add_entry( avg_time * nb_steps, (avg_time/overall_avg)*100.0, pop.name+"_"+func, "cpp core")

        # single operations for projections
        for proj in network.get_projections():
            for func in ["psp", "step", "post_event"]:
                avg_time = Profiler()._cpp_profiler.get_avg_time(proj.name, func)
                Profiler().add_entry( avg_time * nb_steps, (avg_time/overall_avg)*100.0, proj.name+"_"+func, "cpp core")

        monitor_avg = Profiler()._cpp_profiler.get_avg_time("network", "record")
        Profiler().add_entry( monitor_avg * nb_steps, (monitor_avg/overall_avg)*100.0, "record", "cpp core")

def simulate_until(max_duration:float, population: Population | list[Population], operator='and', measure_time:bool = False, net_id:int=0):
    """
    Runs the network for the maximal duration in milliseconds. If the ``stop_condition`` defined in the population becomes true during the simulation, it is stopped.

    One can specify several populations. If the stop condition is true for any of the populations, the simulation will stop ('or' function).

    Returns the actual duration of the simulation in milliseconds.

    Example:

    ```python
    pop1 = ann.Population( ..., stop_condition = "r > 1.0 : any")
    ann.compile()
    duration = ann.simulate_until(max_duration=1000.0, population=pop1)
    ```

    :param max_duration: Maximum duration of the simulation in milliseconds.
    :param population: (list of) population(s) whose ``stop_condition`` should be checked to stop the simulation.
    :param operator: Operator to be used ('and' or 'or') when multiple populations are provided (default: 'and').
    :param measure_time: Defines whether the simulation time should be printed (default=False).
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)

    # Sanity checks
    if not network.compiled:
        Messages._error('simulate_until(): the network is not compiled yet.')
    if not network.instance:
        Messages._error('simulate_until(): the network is not initialized yet.')

    # Compute maximum number of steps
    nb_steps = ceil(float(max_duration) / ConfigManager().get("dt", net_id))

    if not isinstance(population, list):
        population = [population]

    # Perform the simulation until max_duration is reached or the conditio is fulfilled.
    if measure_time:
        tstart = time.time()

    nb = network.instance.run_until(nb_steps, [pop.id for pop in population], True if operator=='and' else False)

    sim_time = float(nb) / ConfigManager().get("dt", net_id)
    if measure_time:
        Messages._print('Simulating', nb/ConfigManager().get("dt", net_id)/1000.0, 'seconds of the network took', time.time() - tstart, 'seconds.')
    return sim_time

def step(net_id=0):
    """
    Performs a single simulation step (duration = `dt`).
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)

    # Sanity check
    if not network.compiled:
        Messages._error('step(): the network is not compiled yet.')
    if not network.instance:
        Messages._error('step(): the network is not initialized yet.')

    # Simulate a single step
    network.instance.step()


################################
## Decorators
################################

def callbacks_enabled(net_id=0):
    """
    Returns True if callbacks are enabled for the network.
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)
    return network._callbacks_enabled

def disable_callbacks(net_id=0):
    """
    Disables all callbacks for the network.
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)
    network._callbacks_enabled = False

def enable_callbacks(net_id=0):
    """
    Enables all declared callbacks for the network.
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)
    network._callbacks_enabled = True

def clear_all_callbacks(net_id=0):
    """
    Clears the list of declared callbacks for the network.

    Cannot be undone!
    """
    # Access the network
    network = NetworkManager().get_network(net_id=net_id)
    network._callbacks.clear()


class every :
    """
    Decorator to declare a callback method that will be called periodically during the simulation.

    Example of setting increasing inputs to a population every 100 ms, with an offset of 90 ms (or -10 ms relative to the period):

    ```python
    net = ann.Network()

    @ann.every(network=net, period=100., offset=-10.)
    def step_input(n):
        pop.I = float(n) / 100.

    net.simulate(10000.)
    ```

    ``step_input()`` will be called at times 90, 190, ..., 9990 ms during the call to ``simulate()``.

    The method must accept only ``n`` as parameter (an integer being 0 the first time the method is called, and incremented afterwards) and can not return anything.

    The times at which the method is called are relative to the time when ``simulate()`` is called (if ``t`` is already 150 before calling ``simulate()``, the first call will then be made at ``t=240`` with the previous example).

    If multiple callbacks are defined, they will be called in the order of their declaration if they occur at the same time.

    ``wait`` can be combined with ``offset``, so if ``period=100.``, ``offset=50.`` and ``wait=500.``, the first call will be made 550 ms after the call to ``simulate()`
    
    :param network: the network instance that will catch the callbacks. By default it is the top-level network of id 0.
    :param period: interval in ms between two calls to the function. If less than ``dt``, will be called every step.
    :param offset: by default, the first call to the method will be made at the start of the simulation. The offset delays the call within the period (default: 0.0). Can be negative, in which case it will be counted from the end of the period.
    :param wait: allows to wait for a certain amount of time (in ms) before starting to call the method.

    """

    def __init__(self, network:"Network"=None, period:float=1.0, offset:float=0., wait:float=0.0) -> None:

        self.network = network if network is not None else NetworkManager().magic_network()
        self.network._callbacks.append(self)

        self.period = max(float(period), self.network.dt)
        self.offset = min(float(offset), self.period)
        self.wait = max(float(wait), 0.0)

    def __call__(self, f):
        
        # If there are decorator arguments, __call__() is only called
        # once, as part of the decoration process! You can only give
        # it a single argument, which is the function object.
        self.func = f
        return f


def _simulate_with_callbacks(duration, net_id=0):
    """
    Replaces simulate() when call_backs are defined.
    """

    # Access the network
    network = NetworkManager().get_network(net_id=net_id)

    # Duration
    t_start = Global.get_current_step(net_id)
    length = int(duration/network.dt)

    # Compute the times
    times = []
    for c in network._callbacks:
        period = int(c.period/network.dt)
        offset = int(c.offset/network.dt) % period
        wait = int(c.wait/network.dt)

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
            network.instance.run(time-Global.get_current_step(net_id))
        # Call the callback
        callback.func(n)

    # Go to the end of the duration
    if Global.get_current_step(net_id) < t_start + length:
        network.instance.run(t_start + length - Global.get_current_step(net_id))
