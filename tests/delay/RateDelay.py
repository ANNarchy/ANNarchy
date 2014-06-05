#
#    A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1)
#setup( paradigm = "cuda", num_threads=1 )

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        r = t / 10.0 : init = 0.0 
    """
)

OutputNeuron = RateNeuron(
    equations="""
        r = sum(exc)
    """
)

input_pop = Population(geometry=(10), neuron=InputNeuron)
output_pop = Population(geometry=(1), neuron=OutputNeuron)

input_output = Projection( input_pop, output_pop, 'exc').connect_all_to_all(1.0, 2)

compile()

simulate(5)