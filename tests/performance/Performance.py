#
#	A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1)
#setup( paradigm = "cuda", num_threads=1 )

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

NEURON = 15
CONN = 10

input_pop = Population(geometry=(NEURON), neuron=InputNeuron)
output_pop = Population(geometry=(1), neuron=OutputNeuron)

proj = Projection(input_pop, output_pop, 'exc').connect_fixed_number_pre(CONN)

compile() # needed to save connectivity matrix
#proj.save_connectivity_as_csv()

simulate(1)

print 'values:'
proj._cython_instance._get_value(0)

# recompile as stand alone
#compile(cpp_stand_alone = True)
