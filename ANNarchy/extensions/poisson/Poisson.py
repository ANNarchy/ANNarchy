from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import SpikeNeuron

class PoissonPopulation(Population):

    def __init__(self, geometry, name=None, rates=10.0):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. 

            * *name*: unique name of the population (optional).

            * *rates*: mean firing rate of each neuron (default: 10.0 Hz)
        
        """  
        poisson_neuron = SpikeNeuron(
            parameters = """
            rates = %(rates)s
            """ % {'rates': rates},
            equations = """
            p = Uniform(0.0, 1.0) * 1000.0 / dt
            """,
            spike = """
                p <= rates
            """

        )
        Population.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)