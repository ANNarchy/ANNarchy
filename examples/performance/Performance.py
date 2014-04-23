#
#	A network comparable to Dinkelbach et al. 2012
#
from ANNarchy4 import *

setup( paradigm = "cuda" )

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        rate = 0.2 : init = 0.1 
    """
)

OutputNeuron = RateNeuron(
    equations="""
        rate = sum(exc)
    """
)

NEURON = 1000
CONN = 10

input_pop = Population(geometry=(NEURON), neuron=InputNeuron)
output_pop = Population(geometry=(NEURON), neuron=OutputNeuron)

proj = Projection(input_pop, output_pop, 'exc').connect_fixed_number_pre(CONN)

compile() # needed to save connectivity matrix
proj.save_connectivity_as_csv()

# recompile as stand alone
compile(cpp_stand_alone = True)