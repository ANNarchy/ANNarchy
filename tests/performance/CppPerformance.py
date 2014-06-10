#
#	 Implementation of the performance profiling as presented in Dinkelbach et al. 2012
#
from ANNarchy import *
from ANNarchy.extensions.Profile import *

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        r = 0.2 : init = 0.1 
    """
)

OutputNeuron = RateNeuron(
    equations="""
        r = sum(exc)
    """
)

# setup net        
input_pop = Population(geometry=(1000), neuron=InputNeuron)
output_pop = Population(geometry=(20), neuron=OutputNeuron)

input_output = Projection( input_pop, output_pop, 'exc').connect_fixed_number_pre(100, 1.0)

compile()

# run steps
simulate(100)
