#
#   A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1, structural_plasticity=True)

# Defining the neurons
InputNeuron = Neuron(
    parameters="""
        r = 0.0
    """

)

OutputNeuron = Neuron(
    equations="""
        r = sum(exc)
        200.0*dtrace/dt + trace = r
    """
)

DefaultSynapse = Synapse(
    parameters = "max_age = 10.0 : postsynaptic",
    equations = "age = if pre.r > 0.5: 0.0 else: age + dt; proba_pruning = Uniform(0.0, 1.0): init=1.0",
    pruning="(age > max_age) : proba=0.1",
    creating="(pre.r > 0.5) : w=0.5"
)


NEURON = 5

input_pop = Population(geometry=NEURON, neuron=InputNeuron)
output_pop = Population(geometry=1, neuron=OutputNeuron)

proj = Projection(input_pop, output_pop, 'exc', synapse = DefaultSynapse).connect_from_matrix(weights=[[1.0, 1.0, 0.0, None, None]])

compile() 


input_pop[0].r = 1.0
input_pop[4].r = 1.0



print '3 synapses initially'
print 'ranks:', proj[0].pre_rank
print 'weights:', proj[0].w
proj.start_creating()
simulate(1.0)
print 'A synapse to neuron 4 should be added.'
print 'ranks:', proj[0].pre_rank
print 'weights:', proj[0].w
proj.stop_creating()

print 'Simulating again is not a problem'
simulate(10.0)
print 'ranks:', proj[0].pre_rank
print 'weights:', proj[0].w

print 'But it crashes on exit...'