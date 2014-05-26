#
#	Implementation of the performance profiling as presented in Dinkelbach et al. 2012
#
from ProfileNetwork import ProfileNetwork
from ANNarchy.extensions.Profile import *

#
# test SETUP
START_NUMBER_OF_NEURONS = 10
END_NUMBER_OF_NEURONS = 1000

START_NUMBER_OF_CONNECTIONS = 10
END_NUMBER_OF_CONNECTIONS = 1000

n = START_NUMBER_OF_NEURONS
neuron_config = []
while n < END_NUMBER_OF_NEURONS:
    neuron_config.append(n)
    n *= 2 

c = START_NUMBER_OF_CONNECTIONS
connection_config = []
while c < END_NUMBER_OF_CONNECTIONS:
    connection_config.append(c)
    c *= 2 

print neuron_config
print connection_config

for neur in neuron_config: 
    
    for conn in connection_config:
        
        net = ProfileNetwork(neur, conn)
        
        print 'Script: compile'
        net.compile()
    
        print 'Script: simulate'
        net.simulate(1000)
        
        print 'Script: destroy'
        net.destroy()