from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Record import Monitor
import Global
import ANNarchy.generator.Generator as Generator

import os, shutil, sys

class Network(object):
    """ 
    A network gathers already defined populations and projections to be run independently.
    """
    def __init__(self, magic=False):
        """
        *Parameters:*
        * **magic**: defines if all existing populations and projections should be automatically added (default: false)
        """
        self.id = len(Global._network)
        self.magic = magic
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
        if magic:
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
        if True:#not self.magic: # Complete compilation
            Generator.compile(directory=directory, silent=silent, net_id=self.id)
        else:
            if not Global._network[0]['compiled']: # Compile the magic network
                Generator.compile(directory=directory, silent=silent, net_id=0)
            subdir = 'annarchy/net'+str(self.id)+'/'
            try:
                os.makedirs(subdir)
            except:
                pass # directory already existed
            shutil.copy('annarchy/ANNarchyCore0.so', subdir)
            Generator._instantiate(net_id=self.id, single=False)

    def simulate(self, duration):
        "Simulates the network for the given duration in ms."
        Global.simulate(duration, net_id=self.id)


def parallel_run(method, networks=None, number=0, measure_time=False, sequential=False):
    """
    Allows to run multiple networks in parallel using multiprocessing.

    If the ``networks`` argument is provided as a list of Network objects, the given method will be executed for each of these networks.

    If ``number`` is given instead, the same number of magic networks will be created and the method is applied. The method returns in this case a list of networks.

    *Parameters*:

    * **method**: a Python method which will be executed for each network. This function must accept a Network object as a first argument.
    * **networks**: a list of networks to simulate in parallel.
    * **measure_time**: if the total simulation time should be printed out. 
    * **sequential**: runs the networks sequentially instead of in parallel. 
    """
    # Check inputs
    if not networks and number < 1:
        Global._error('parallel_run(): the networks or number arguments must be set.')
        return []
    if not networks: # The magic network will run N times
        networks = []
        for i in range(number):
            net = Network(True)
            net.compile(silent=True)
            networks.append(net)
    if not isinstance(networks, list):
        Global._error('parallel_run(): the networks argument must be a list.')
        return []
    import types
    if not isinstance(method, types.FunctionType):
        Global._error('parallel_run(): the method argument must be a method.')
        return []

    # Number of networks
    nb_nets = len(networks)

    # Time measurement    
    from time import time
    if measure_time:
        ts = time()

    # Simulation
    if not sequential:
        from multiprocessing.dummy import Pool
        pool = Pool(nb_nets)
        try:
            results = pool.map(method, networks)
        except:
            Global._error('parallel_run(): running multiple networks failed.')
            return []
        pool.close()
        pool.join()
    else:
        results = []
        for net in networks:
            try:
                results.append(method(net))
            except:
                Global._error('parallel_run(): running network ' + str(net.id) + ' failed.')
                return []

    # Time measurement
    if measure_time:
        msg = 'Running ' + str(nb_nets) + ' networks'
        if not sequential:
            msg += ' in parallel '
        else:
            msg += ' sequentially '
        msg += 'took: ' + str(time()-ts)
        Global._print(msg)

    if number > 0:
        return networks, results
    return results