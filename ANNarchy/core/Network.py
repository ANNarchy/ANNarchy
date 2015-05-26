from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Record import Monitor
import Global
import ANNarchy.generator.Generator as Generator

import os, shutil, sys
import numpy as np

class Network(object):
    """ 
    A network gathers already defined populations and projections to be run independently.
    """
    def __init__(self, everything=False):
        """
        *Parameters:*
        * **everything**: defines if all existing populations and projections should be automatically added (default: false)
        """
        self.id = len(Global._network)
        self.everything = everything
        Global._network.append(
            {
            'populations': [],
            'projections': [],
            'monitors': [],
            'instance': None,
            'compiled': False
            }
        )
        self.populations = []
        self.projections = []
        self.monitors = []
        if everything:
            self.add(Global._network[0]['populations'])
            self.add(Global._network[0]['projections'])
            self.add(Global._network[0]['monitors'])

    def add(self, objects):
        """
        Adds a Population, Projection or Monitor to the network.

        *Parameters:*
        * **objects**: A single object or a list to add to the network.
        """
        if isinstance(objects, list):
            for item in objects:
                self._add_object(item)
        else:
            self._add_object(objects)

    def _add_object(self, obj):
        if isinstance(obj, Population):
            # Create a copy
            pop = Population(geometry=obj.geometry, neuron=obj.neuron_type, name=obj.name, stop_condition=obj.stop_condition)
            # Remove the copy from the global network
            Global._network[0]['populations'].pop(-1)
            # Copy import properties
            pop.id = obj.id
            pop.name = obj.name
            pop.class_name = obj.class_name
            pop.init = obj.init
            # Add the copy to the local network
            Global._network[self.id]['populations'].append(pop)
            self.populations.append(pop)

        elif isinstance(obj, Projection):
            # Check the pre- or post- populations
            try:
                pre_pop = self.get(obj.pre)
                if isinstance(obj.pre, PopulationView):
                    pre = PopulationView(pre_pop, obj.pre.ranks)
                else:
                    pre = pre_pop
                post_pop = self.get(obj.post)
                if isinstance(obj.post, PopulationView):
                    post = PopulationView(post_pop, obj.post.ranks)
                else:
                    post = post_pop
            except:
                Global._error('Network.add(): The pre- or post-synaptic population of this projection are not in the network.')
                exit(0)
            target = obj.target
            synapse = obj.synapse
            # Create the projection
            proj = Projection(pre=pre, post=post, target=target, synapse=synapse)
            # Remove the copy from the global network
            Global._network[0]['projections'].pop(-1)
            # Copy import properties
            proj.id = obj.id
            proj.name = obj.name
            proj.init = obj.init
            # Copy the synapses if they are already created
            proj._store_csr(obj._get_csr()) 
            # Add the copy to the local network
            Global._network[self.id]['projections'].append(proj)
            self.projections.append(proj)
            
        elif isinstance(obj, Monitor):
            m = Monitor(obj.object, variables=obj.variables, period=obj._period, start=obj._start, net_id=self.id)
            # Add the copy to the local network (the monitor writes itself already in the right network)
            self.monitors.append(m)


    def get(self, obj):
        """
        Returns the local Population, Projection or Monitor identical to the provided argument.

        *Parameters:*
        * **obj**: A single object.

        **Example**::

            pop = Population(100, Izhikevich)
            net = Network()
            net.add(pop)
            net.compile()
            net.simulate(100.)
            print net.get(pop).v
        """
        if isinstance(obj, (Population, PopulationView)):
            for pop in self.populations:
                if pop.id == obj.id:
                    return pop
        elif isinstance(obj, Projection):
            for proj in self.projections:
                if proj.id == obj.id:
                    return proj
        elif isinstance(obj, Monitor):
            for m in self.monitors:
                if m.id == obj.id:
                    return m
        Global._error('The network has no such object.')
        return None

    def compile(self, directory='annarchy', silent=False):
        """Compiles the network.

        *Parameters*:

        * **directory**: name of the subdirectory where the code will be generated and compiled.
        * **silent**: defines if the "Compiling... OK" should be printed.
        """
        Generator.compile(directory=directory, silent=silent, net_id=self.id)

    def simulate(self, duration):
        "Simulates the network for the given duration in ms."
        Global.simulate(duration, net_id=self.id)


def parallel_run(method, networks=None, number=0, measure_time=False, sequential=False):
    """
    Allows to run multiple networks in parallel using multiprocessing.

    If the ``networks`` argument is provided as a list of Network objects, the given method will be executed for each of these networks.

    If ``number`` is given instead, the same number of networks will be created and the method is applied. 

    **Returns**:

    * a list of the values returned by ``method``.

    If ``number`` is used, the created networks are not returned, you should return what you need to analyse.

    *Parameters*:

    * **method**: a Python method which will be executed for each network. This function must accept an integer as first argument (id of the simulation) and a Network object as second argument.
    * **networks**: a list of networks to simulate in parallel.
    * **measure_time**: if the total simulation time should be printed out. 
    * **sequential**: if True, runs the simulations sequentially instead of in parallel (default: False). 
    """
    # Check inputs
    if not networks and number < 1:
        Global._error('parallel_run(): the networks or number arguments must be set.')
        return []

    import types
    if not isinstance(method, types.FunctionType):
        Global._error('parallel_run(): the method argument must be a method.')
        return []

    if not networks: # The magic network will run N times
        return _parallel_multi(method, number, measure_time, sequential)

    if not isinstance(networks, list):
        Global._error('parallel_run(): the networks argument must be a list.')
        return []

    # Simulate the different networks
    return _parallel_networks(method, networks, measure_time, sequential)

def _parallel_networks(method, networks, measure_time, sequential):
    " Method when different networks are provided"
    import multiprocessing
    from multiprocessing.dummy import Pool

    # Time measurement    
    from time import time
    if measure_time:
        ts = time()

    # Simulation
    if not sequential:
        pool = Pool(multiprocessing.cpu_count())
        try:
            results = pool.map(_only_run_method, [(idx, method, net) for idx, net in enumerate(networks)])
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running multiple networks failed.')
            return []
        pool.close()
        pool.join()
    else:
        results = []
        for idx, net in enumerate(networks):
            try:
                results.append(method(idx, net))
            except Exception as e:
                Global._print(e)
                Global._error('parallel_run(): running network ' + str(net.id) + ' failed.')
                return []

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(len(networks)) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Global._print(msg)

    return results


def _parallel_multi(method, number, measure_time, sequential):
    "Method when the same network must be simulated multiple times."
    import multiprocessing
    from multiprocessing import Pool

    # Time measurement    
    from time import time
    if measure_time:
        ts = time()

    # Simulation
    if not sequential:
        try:
            pool = Pool(multiprocessing.cpu_count())
            results = pool.map(_create_and_run_method, [(n, method) for n in range(number)] )
            pool.close()
            pool.join()
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running ' + str(number) + ' networks failed.')
            return []
    else:
        results = []
        try:
            for n in range(number):
                results.append(_create_and_run_method((n, method)))
        except Exception as e:
            Global._print(e)
            Global._error('parallel_run(): running ' + str(number) + ' networks failed.')
            return []

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(number) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Global._print(msg)
        
    return results

def _create_and_run_method(args):
    "method called to wrap the user-defined method"
    n, method = args
    net = Network(True)
    np.random.seed() # TODO: if seed is declared
    Generator._instantiate(net.id, 0)
    res = method(n, net)
    del net
    return res

def _only_run_method(args):
    "method called to wrap the user-defined method"
    n, method, net = args
    res = method(n, net)
    return res