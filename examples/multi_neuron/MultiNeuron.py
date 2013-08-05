#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *

#
# Define the neuron classes
#
Input = Neuron(   tau = 1.0,
                  rate = Variable(init = 0.0)
               )

Layer1 = Neuron(   tau = 10.0,
                   mp = Variable(init=0.0, eq="tau * dmp / dt + mp = sum(exc)"),
                   rate = Variable(init=0.0, eq="rate=pos(mp)"),
                   order = ['mp','rate']
	      )

Layer2 = Neuron(   tau = 10.0,
                   mp = Variable(init=0.0, eq="tau * dmp / dt + mp = sum(exc) - sum(inh) "),
                   rate = Variable(init=0.0, eq="rate=pos(mp)"),
                   order = ['mp','rate']
              )

Oja = Synapse(tau = 5000,
              dt = 1.0,
              alpha = Variable(init=0.1, eq = "alpha = post.rate"),
              value = Variable(init=0.0, eq = "dvalue / dt = alpha * (pre.rate*post.rate - post.rate*post.rate*value)"),
              psp = Variable(eq = "psp=(1-pre.rate)*value")

              )

InputPop = Population("Input", (8,8,1), Input)
Layer1Pop = Population("Layer1", (8,8,1), Layer1)
Layer2Pop = Population("Layer2", (6,5,1), Layer2)

Proj_In_L1 = Projection(pre="Input", post="Layer1", target='exc', connector=Connector('One2One', weights=RandomDistribution('constant', [1.0])))
Proj_L1_L2 = Projection(pre="Layer1", post="Layer2", target='exc', synapse=Oja, connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,0.1])))
Proj_L2_L2 = Projection(pre="Layer2", post="Layer2", target='inh', connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,0.1])))

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

import math
import numpy as np

if __name__ == '__main__':

    #plotData = [{'pop': InputPop, 'var': 'rate', 'name':'input.rate'}, {'pop': Layer1Pop, 'var': 'rate', 'name':'layer1.rate'}]
    #PlotThread(plotData, True)

    print 'Running the simulation'

    for trial in range(5000):
        bars = np.zeros((8,8))

        for i in xrange(8):
            # appears a horizontal bar?
            if np.random.rand(1) < 1.0/8.0:
               bars[:,i] = 1.0

            # appears a vertical bar?
            if np.random.rand(1) < 1.0/8.0:
               bars[i,:] = 1.0

        InputPop.cyInstance.rate = bars.reshape(8*8)
       
        simulate(1000)

        print Layer1Pop.cyInstance.rate
