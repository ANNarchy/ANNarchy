#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *
from datetime import datetime

setup(suppress_warnings=True)

#
# Define the neuron classes
#
Input = Neuron(   
    rate = Variable(init = 0.0),
)

Output = Neuron(   
    tau = 1.0,
    mp = Variable(init = 0, eq="mp = sum(exc)"),
    rate = Variable(init = 0.0, eq="tau * drate/dt +rate = pos(mp)"),
    order = ['mp','rate']
)

InputPop = Population(name="Input", geometry=(8,1), neuron=Input)
Layer1Pop = Population(name="Layer1", geometry=(8,1), neuron=Output)

Proj = Projection(pre="Input", post="Layer1", target='exc', connector=One2One(weights=1.0, delays=2.0))

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

def synapse_test():

    InputPop.rate = Uniform(0,1).get_values(InputPop.geometry)

    for i in range(5):
        simulate(1)
        print Layer1Pop.rate
        #print InputPop.rate

    #print InputPop.rate
    #print Layer1Pop.rate

if __name__ == '__main__':

    synapse_test()
