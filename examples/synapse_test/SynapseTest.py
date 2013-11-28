#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *
from datetime import datetime

#
# Define the neuron classes
#
Simple = Neuron(  tau = 1.0,
                  baseline = Variable(init = 1, type=int),
                  rate = Variable(init = 0.0, max=1.5, type=float, eq="rate = baseline")
               )

SimpleSynapse = Synapse(
    tau = 1.0,
    boolPar = True,
    boolVar = Variable(init=True),
    intVar = Variable(init=1, type=int),
    value2 = Variable(init=0, eq="value2 = 1.0 / pre.rate", min=-0.5, max=1.0, type=float),
    value = Variable(init=0, eq="value = 1.0 / pre.rate", min=-0.5, max=1.0)
)

InputPop = Population(name="Input", geometry=(8,8), neuron=Simple)
Layer1Pop = Population(name="Layer1", geometry=(1,1), neuron=Simple)

Proj = Projection(pre="Input", post="Layer1", target='exc', synapse=SimpleSynapse, connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,1.0])))

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

import math
import numpy as np

if __name__ == '__main__':

    #print 'target:',Proj.dendrite[0].target

    #
    #   synapse removal
    print Proj.dendrite(0).rank
    print Proj.dendrite(0).value
    
    print 'test: remove synapse to neuron 1'
    print 'result', Proj.dendrite(0).remove_synapse(1)
    
    print Proj.dendrite(0).rank
    print Proj.dendrite(0).value
    
    #
    #   synapse add - this test should lead to an error output
    print 'remove synapse to neuron 1'    
    Proj.dendrite(0).remove_synapse(1)
    
    #
    #   synapse add - this test should lead to an error output
    print 'add synapse (0, , 0.11111, 0)'    
    Proj.dendrite(0).add_synapse(0, 0.11111, 0)
    
    #
    #   synapse add
    print 'add synapse (1, 0.11111, 0)'        
    Proj.dendrite(0).add_synapse(1, 0.11111, 0)

    print Proj.dendrite(0).rank
    print Proj.dendrite(0).value
    
