#
#	Implementation of the performance profiling as presented in Dinkelbach et al. 2012
#
from ProfileNetwork import ProfileNetwork
from ANNarchy.extensions.Profile import *

#
# test SETUP
#===============================================================================
# START_NUMBER_OF_NEURONS = 800
# END_NUMBER_OF_NEURONS = 51200
# 
# START_NUMBER_OF_CONNECTIONS = 100
# END_NUMBER_OF_CONNECTIONS = 100
#===============================================================================

START_NUMBER_OF_NEURONS = 2000
END_NUMBER_OF_NEURONS = 2000

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

thread_config = [1,2,4]

print neuron_config
print connection_config
print thread_config

for neur in neuron_config: 
    
    neuron_scalability = Scalability(['sum'], connection_config, thread_config, 3)
    
    for conn in connection_config:
        
        if conn > neur:
            continue

        # setup net        
        net = ProfileNetwork(neur, conn)
        net.compile()

        # setup profiler
        profiler = Profile(thread_config, 50, 'profile_'+str(neur)+'_'+str(conn), 'tests')
        profiler.add_to_profile(net)
        
        # profiling ...
        profiler.measure_func(net.simulate, 10)
        
        profiler.analyse_data()
        
        profiler.save_to_file()
        
        neuron_scalability.add_data_set('sum', conn, profiler._pop_data['Population1']['sum'].mean())
        net.destroy()
        
        profiler.print_data()
        
    neuron_scalability.analyze_data()
    
    neuron_scalability.visualize_data()
    
    raw_input()