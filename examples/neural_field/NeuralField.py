#
#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *

#
# Define the neuron classes
#
Input = Neuron(   tau = 1.0,
                  noise = 0.5,
                  baseline = Variable(init = 0.0),
                  mp = Variable(eq = "tau * dmp / dt + mp = baseline + noise * (2 * RandomDistribution('uniform', [0,1]) - 1)"),
                  rate = Variable(eq = "rate = pos(mp)"),
                  order = ['mp','rate'] 
               )

Focus = Neuron( tau = 20.0,
                noise = 0.0,
                threshold_min = 0.0,
                threshold_max = 1.0,
                rate = Variable(init = 0.0, eq = 'rate = sum(exc)')
                #mp = Variable(eq = "tau * dmp / dt + mp = sum(exc) + sum(inh) + baseline + noise * (2 * RandomDistribution('uniform', [0,1]) - 1)")
	       )
		
				
InputPop = Population("Input", (20,20,1), Input)
FocusPop = Population("Focus", (20,20,1), Focus)

Proj1 = Projection( pre = InputPop, post = "Focus", target = 'exc', connector = Connector( type='One2One', weights=RandomDistribution('uniform', [0,1]) ) )
Proj2 = Projection( pre = "Focus", post = "Focus", target = 'inh', connector = Connector( type='DoG', weights=RandomDistribution('uniform', [0,1]) ) )

#
# Analyse and compile everything, initialize the parameters/variables...
#
Compile()

import math
import numpy as np

#
# Main program
#
if __name__ == "__main__":

    plotData = [{'pop': InputPop, 'var': 'baseline', 'name': 'baseline'}, {'pop': InputPop, 'var': 'rate', 'name':'input.rate'}, {'pop': FocusPop, 'var': 'rate', 'name': 'focus.rate'}]
 
    w = 20
    h = 20
    data = np.zeros((1,w*h))
    
    angle = 0.0
    radius = 0.5
    sigma = 2.0

    PlotThread(plotData, True)

    for i in range (10000):

        angle += 1.0/5000.0

        cw = w / 2.0 * ( 1.0 + radius * math.cos(2 * math.pi * angle ) )
        ch = h / 2.0 * ( 1.0 + radius * math.sin(2 * math.pi * angle ) )

        for x in xrange(w):
            for y in xrange(h):
                dist = (x-cw)*(x-cw)+(y-ch)*(y-ch)
                value = 0.5 * math.exp(-dist/2.0/sigma/sigma)
                idx = x+y*w
                data[(0,idx)] = value

        InputPop.cyInstance.baseline = data[0]
        Simulate(10)
