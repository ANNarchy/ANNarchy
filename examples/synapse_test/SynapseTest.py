#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *
from datetime import datetime

#
# Define the neuron classes
#
Simple = Neuron(   tau = 1.0,
                  rate = Variable(init = 0.0)
               )

InputPop = Population("Input", (8,8,1), Simple)
Layer1Pop = Population("Layer1", (1,1,1), Simple)

Proj = Projection(pre="Input", post="Layer1", target='exc', connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,1.0])))

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

import math
import numpy as np

if __name__ == '__main__':

    print 'target:',Proj.local_projections[0].target

    #
    #   synapse removal
    print Proj.local_projections[0].rank
    print Proj.local_projections[0].value
    
    print 'remove synapse to neuron 1'
    Proj.local_projections[0].remove_synapse(1)
    
    print Proj.local_projections[0].rank
    print Proj.local_projections[0].value
    
    #
    #   synapse add - this test should lead to an error output
    print 'remove synapse to neuron 1'    
    Proj.local_projections[0].remove_synapse(1)
    
    #
    #   synapse add - this test should lead to an error output
    print 'add synapse (0, , 0.11111, 0)'    
    Proj.local_projections[0].add_synapse(0, 0.11111, 0)
    
    #
    #   synapse add
    print 'add synapse (1, 0.11111, 0)'        
    Proj.local_projections[0].add_synapse(1, 0.11111, 0)

    print Proj.local_projections[0].rank
    print Proj.local_projections[0].value
    
