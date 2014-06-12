#
#	 Implementation of the performance profiling as presented in Dinkelbach et al. 2012
#
from ProfileNetwork import ProfileNetwork
from ANNarchy.extensions.Profile import *

#
# test SETUP
#

#===============================================================================
# START_NUMBER_OF_NEURONS = 800
# END_NUMBER_OF_NEURONS = 51200
# 
# START_NUMBER_OF_CONNECTIONS = 100
# END_NUMBER_OF_CONNECTIONS = 100
#===============================================================================

START_NUMBER_OF_NEURONS = 4096
END_NUMBER_OF_NEURONS = 4096

START_NUMBER_OF_CONNECTIONS = 16
END_NUMBER_OF_CONNECTIONS = 4096

n = START_NUMBER_OF_NEURONS
neuron_config = []
while n <= END_NUMBER_OF_NEURONS:
    neuron_config.append(n)
    n *= 2 

c = START_NUMBER_OF_CONNECTIONS
connection_config = []
while c <= END_NUMBER_OF_CONNECTIONS:
    if c > END_NUMBER_OF_NEURONS:
        break
        
    connection_config.append(c)
    c *= 2 

thread_config = [1,2,3,4,5,6,7,8]
operation = 'sum'

#
# test overview
#
print 'running the following test setup:'
print 'neur_config', neuron_config
print 'conn_config', connection_config
print 'thread_config', thread_config

#
# scalability plot is over all tests
scalability = Scalability([operation], connection_config, thread_config, len(thread_config))

#
# run now the tests
for neur in neuron_config: 
    
    for conn in connection_config:
        
        if conn > neur:
            continue

        # setup net        
        net = ProfileNetwork(neur, conn)
        net.compile()

        #
        # setup profiler for one network each
        profiler = Profile(thread_config, 100, 'profile_'+str(neur)+'_'+str(conn), 'tests')
        profiler.add_to_profile(net)
        
        # profiling ...
        profiler.measure_func(net.simulate, 1)

        # needed for getting the data        
        profiler.analyse_data()
        
        scalability.add_data_set(operation, conn, profiler._pop_data['Population1'][operation].mean())
        net.destroy()

#
# evaluate scalability
scalability.analyze_data()

#scalability.visualize_data()

scalability.save_as_mat()

raw_input()