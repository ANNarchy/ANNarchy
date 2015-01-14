#
#	 Implementation of the performance profiling as presented in Dinkelbach et al. 2012
#
from ANNarchy import *
from ANNarchy.core import Global

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        r = Uniform(0,1)
    """
)

OutputNeuron = RateNeuron(
    equations="""
        r = sum(exc)
    """
)

# setup net        
input_pop = Population(geometry=(10000), neuron=InputNeuron, name="Input")
output_pop = Population(geometry=(20), neuron=OutputNeuron, name="Output")

input_output = Projection( input_pop, output_pop, 'exc').connect_fixed_number_pre(1000, 0.1)

compile()

# run steps
for i in range(1,7):
    Global._network.set_number_threads(i)
    simulate(200)
