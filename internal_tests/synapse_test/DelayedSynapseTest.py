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
Input = RateNeuron(   
    parameters="""
        rate = 0.1
    """
)

Output = RateNeuron(   
    parameters="""
        tau = 1.0,
    """,
    equations="""
        mp1 = sum(exc)
        mp2 = sum(exc2)
        tau * drate/dt +rate = pos(mp1) + pos(mp2)
    """
)

InputPop = Population(name="Input", geometry=(8,1), neuron=Input)
Layer1Pop = Population(name="Layer1", geometry=(8,1), neuron=Output)

Proj = Projection(pre="Input", post="Layer1", target='exc').connect_one_to_one(weights=1.0, delays=2.0)
Proj2 = Projection(pre="Input", post="Layer1", target='exc2').connect_all_to_all(weights=1.0, delays=2.0)

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

def synapse_test():

    InputPop.rate = Uniform(0,1).get_values(InputPop.geometry)

    for i in range(5):
        print i 
        simulate(1)

    print InputPop.rate
    print Layer1Pop.rate

if __name__ == '__main__':

    print 'xxx'
    synapse_test()
