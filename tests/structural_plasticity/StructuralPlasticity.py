#
#	A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1, structural_plasticity=True)

# Defining the neurons
InputNeuron = Neuron(
    equations="""
        r += 0.2 : init = 0.0
    """,
    spike="r>0.1",
    reset="r=0.0"

)

OutputNeuron = Neuron(
    equations="""
        r = g_exc
    """,
    spike="r>100.0"
)

DefaultSynapse = Synapse(
    parameters = "toto = 1.0",
    equations = "alpha= 1.0: init=2.0",
    pre_spike = "g_target += w; w += 1.0"
)

NEURON = 30
CONN = 10

input_pop = Population(geometry=(NEURON), neuron=InputNeuron)
output_pop = Population(geometry=(1), neuron=OutputNeuron)

proj = Projection(input_pop, output_pop, 'exc', synapse = DefaultSynapse).connect_fixed_number_pre(CONN, 1.0)

compile() 

proj.alpha = 3.0

print 'Rank:', proj.dendrite(0).rank
print 'Weights:', proj.dendrite(0).w
print 'Alpha:', proj.dendrite(0).alpha

dendrite = proj.dendrite(0)

synapse = dendrite.synapse(proj.dendrite(0).rank[2])

synapse.w = 10.0

for synapse in proj.dendrite(0):
    print synapse.rank, synapse.w

print 'Rank:', proj.dendrite(0).rank
print 'Weights:', proj.dendrite(0).w
print 'Alpha:', proj.dendrite(0).alpha

rank = int(raw_input('Add connection to: '))
proj.dendrite(0).add_synapse(rank, 2.0)


print 'Rank:', proj.dendrite(0).rank
print 'Weights:', proj.dendrite(0).w
print 'Alpha:', proj.dendrite(0).alpha

rank = int(raw_input('Remove connection to: '))

proj.dendrite(0).remove_synapse(rank)

print 'Rank:', proj.dendrite(0).rank
print 'Weights:', proj.dendrite(0).w
print 'Alpha:', proj.dendrite(0).alpha

simulate(10)