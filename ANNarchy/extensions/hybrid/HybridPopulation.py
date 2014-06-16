from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import RateNeuron, SpikeNeuron
import ANNarchy.core.Global as Global
from .Templates import *

class Spike2RatePopulation(Population):
    """
    Converts a population of spiking neurons into a population of rate-coded neurons.

    When building a hybrid network, one need to convert spike trains into an instantaneous firing. Creating a ``Spike2RatePopulation`` allows to get a rate-coded population of the same size as the spiking population.

    Each neuron collects the spikes of corresponding spiking neuron over the last milliseconds (defined by the parameter ``window``), and computes the average firing rate in Hz over this sliding window.

    The firing rate ``r`` of each neuron represents by default the firing rate in Hz. The output can be scaled with the parameter ``scaling``. For example, if you want that ``r=1.0`` represents a firing rate of 100Hz, you can set ``scaling`` to 0.01.

    By definition, the firing rate varies abruptly each time a new spike is perceived. The output can be smoothed with a low-pass filter of time constant ``smooth``.

    .. code-block:: python

        pop2 = Spike2RatePopulation(
            population=pop1, 
            name='rate-coded', 
            window=50.0, 
            smooth=100.0, 
            scaling=0.01
        )
    """
    def __init__(self, population, name=None, window = 100.0, scaling=1.0, smooth=1.0):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be ``RateNeuron``.
        * **name**: the (optional) name of the hybrid population.
        * **window**: the extent of the sliding window (in ms) used to compute the firing rate (default: 100.0 ms).
        * **scaling**: the scaling of the firing rate. Defines what a firing rate of 1 Hz outputs (default: 1.0).
        * **smooth**: time constant (in ms) of the low-pass filter used to smooth the firing rate (default: 1 ms, i.e no smoothing)
        """
        self.population = population
        if not self.population.description['type'] == 'spike':
            Global._error('the population ' + self.population.name + ' must contain spiking neurons.')
            exit(0)

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=name, 
            neuron = RateNeuron(
                parameters="""
                    window = %(window)s : population
                    scaling = %(scaling)s : population
                    smooth = %(smooth)s : population
                """ % {'window': window, 'scaling': scaling, 'smooth': smooth} ,
                equations="r = 0.0"
            ) 
        )
        self.generator = Spike2RatePopulationGenerator(self)

    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, 'py'+ self.class_name)(self.size, self.population._id)
        
        # Create the local attributes and actualize the initial values
        self._init_attributes()

        # Set the global parameters
        self.cyInstance._set_window(self.init['window'])
        self.cyInstance._set_scaling(self.init['scaling'])
        self.cyInstance._set_smooth(self.init['smooth'])

class Rate2SpikePopulation(Population):
    """
    Converts a population of rate-coded neurons into a population of spiking neurons.

    This class allows to generate spike trains based on the computations of a rate-coded network (for example doing visual pre-processing). Creating a ``Rate2SpikePopulation`` allows to get a spiking population of the same size as the rate-coded population.

    The firing rate ``r`` of the rate-coded population represents by default the desired firing rate in Hz. This value can be scaled with the parameter ``scaling``. For example, if you want that ``r=1.0`` represents a firing rate of 100Hz, you can set ``scaling`` to 100.0.

    .. code-block:: python

        pop2 = Rate2SpikePopulation(
            population=pop1, 
            name='spiking', 
            scaling=100.0
        )
    """
    def __init__(self, population, name=None, scaling=1.0):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be ``RateNeuron``.
        * **name**: the (optional) name of the hybrid population.
        * **scaling**: the scaling of the firing rate. Defines what a rate ``r`` of 1.0 means in Hz (default: 1.0).
        """
        self.population = population
        if not self.population.description['type'] == 'rate':
            Global._error('the population ' + self.population.name + ' must contain rate-coded neurons.')
            exit(0)

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=name, 
            neuron = SpikeNeuron(
                parameters="""
                    scaling = %(scaling)s : population
                """ % {'scaling': scaling} ,
                equations="""
                    p = Uniform(0.0, 1.0)
                    rates = p
                """,
                spike="rates>p"
            ) 
        )
        self.generator = Rate2SpikePopulationGenerator(self)

    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, 'py'+ self.class_name)(self.size, self.population._id)
        
        # Create the local attributes and actualize the initial values
        self._init_attributes()

        # Set the global parameters
        self.cyInstance._set_scaling(self.init['scaling'])

class Spike2RatePopulationGenerator(object):
    """ Base class for generating C++ code from a population description. """

    def __init__(self, pop):
        self.pop = pop
        self.class_name = pop.class_name
        self.name = pop.name
        self.id = pop._id
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.class_name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.class_name+'.pyx'
        
        
    def generate(self, verbose):
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(
                s2r_header_template % { 'class_name':self.class_name,
                                    'pre_population': self.pop.population.class_name
                                }
                )
        with open(self.body, mode = 'w') as w_file:
            w_file.write(
                s2r_body_template%{'class_name':self.class_name, 'id': self.id,
                                'pre_population': self.pop.population.class_name}
                )
        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(
                s2r_pyx_template%{'class_name':self.class_name, 'name':self.name,
                                'pre_population': self.pop.population.class_name}
                ) 

class Rate2SpikePopulationGenerator(object):
    """ Base class for generating C++ code from a population description. """

    def __init__(self, pop):
        self.pop = pop
        self.class_name = pop.class_name
        self.name = pop.name
        self.id = pop._id
        
        # Names of the files to be generated
        self.header = Global.annarchy_dir+'/generate/build/'+self.class_name+'.h'
        self.body = Global.annarchy_dir+'/generate/build/'+self.class_name+'.cpp'
        self.pyx = Global.annarchy_dir+'/generate/pyx/'+self.class_name+'.pyx'
        
        
    def generate(self, verbose):
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(
                r2s_header_template % { 'class_name':self.class_name,
                                    'pre_population': self.pop.population.class_name
                                }
                )
        with open(self.body, mode = 'w') as w_file:
            w_file.write(
                r2s_body_template%{'class_name':self.class_name, 'id': self.id,
                                'pre_population': self.pop.population.class_name}
                )
        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(
                r2s_pyx_template%{'class_name':self.class_name, 'name':self.name,
                                'pre_population': self.pop.population.class_name}
                ) 