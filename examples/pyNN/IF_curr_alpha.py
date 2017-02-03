#   ANNarchy - IF_curr_alpha
#
#   Simple network with a Poisson spike source projecting to a pair of IF_curr_alpha neurons.
# 
#   This is a reimplementation of the PyNN example:
#
#   http://www.neuralensemble.org/trac/PyNN/wiki/Examples/simpleNetwork
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay

from ANNarchy import *

# Parameters
dt = 0.1
tstop = 1000.0
rate = 100.0

# Setup
setup(dt=dt)

# Create the Poisson spikes
number = int(2*tstop*rate/1000.0)
np.random.seed(26278342)
spike_times = np.add.accumulate(np.random.exponential(1000.0/rate, size=number))
# assert spike_times.max() > tstop
# print(spike_times.min())

# Input population
input_population  = SpikeSourceArray(list(spike_times))

# Output population
output_population = Population(2, IF_curr_alpha)
output_population.set( { 
    'tau_refrac': 2.0, 
    'v_thresh': -50.0,
    'tau_syn_E': 2.0, 
    'tau_syn_I': 2.0 
})

# Excitatory projection
proj = Projection(input_population, output_population, 'exc')
proj.connect_all_to_all(weights=1.0)

# Compile the network
compile()

# Simulate
m = Monitor(output_population, ['spike', 'v'])
simulate(tstop)
data = m.get()

# Plot the results
import matplotlib.pyplot as plt
plt.plot(dt*np.arange(tstop/dt), data['v'][:, 0])
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.ylim([-66.0, -48.0])
plt.title('Simple Network')
plt.show()


