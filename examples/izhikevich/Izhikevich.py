#
#   ANNarchy - Pulse-coupled network
#
#   Implementation of the pulse-coupled network proposed in:
#
#   Izhikevich, E.M. (2003). Simple Model of Spiking Neurons, IEEE Transaction on Neural Networks, 14:6.
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay
#
from ANNarchy import *
setup(num_threads=8)

# Create the excitatory population
Exc = Population(name='Exc', geometry=800, neuron=Izhikevich)
re = np.random.random(800)
Exc.noise = 5.0
Exc.a = 0.02
Exc.b = 0.2
Exc.c = -65.0 + 15.0 * re**2
Exc.d = 8.0 - 6.0 * re**2
Exc.v = -65.0
Exc.u = Exc.v * Exc.b


# Create the Inh population
Inh = Population(name='Inh', geometry=200, neuron=Izhikevich)
ri = np.random.random(200)
Inh.noise = 2.0
Inh.a = 0.02 + 0.08 * ri
Inh.b = 0.25 - 0.05 * ri
Inh.c = -65.0
Inh.d = 2.0 
Inh.v = -65.0
Inh.u = Inh.v * Inh.b

# Create the projections
exc_exc = Projection(
    pre=Exc, 
    post=Exc, 
    target='exc'
).connect_all_to_all(weights=Uniform(0.0, 0.5))
   
exc_inh = Projection(
    pre=Exc, 
    post=Inh, 
    target='exc',
).connect_all_to_all(weights=Uniform(0.0, 0.5))
  
inh_exc = Projection(
    pre=Inh, 
    post=Exc, 
    target='inh'
).connect_all_to_all(weights=Uniform(0.0, 1.0))
  
inh_inh = Projection(
    pre=Inh, 
    post=Inh, 
    target='inh'
).connect_all_to_all(weights=Uniform(0.0, 1.0))

    
# Compile
compile()

# Start recording the spikes in the network to produce the raster plot
Exc.start_record(['spike', 'v'])
Inh.start_record('spike')

# Simulate 1 second   
simulate(1000000.0, measure_time=True)

# Retrieve the recordings
exc_data = Exc.get_record()
inh_data = Inh.get_record()

# Retrieve the spike timings
spikes_exc = raster_plot(exc_data['spike'])
spikes_inh = raster_plot(inh_data['spike'])
spikes = np.concatenate((spikes_exc, spikes_inh + [0, 800]), axis=0)

# Number of spikes per step in the excitatory population
fr_exc = histogram(exc_data['spike'])

# Plot the results
import pylab as plt
# First plot: raster plot
ax = plt.subplot(3,1,1)
ax.plot(spikes[:, 0], spikes[:, 1], '.', markersize=1.0)
# Second plot: membrane potential of a single excitatory cell
ax = plt.subplot(3,1,2)
ax.plot(exc_data['v']['data'][15, :]) # for example
# Third plot: number of spikes per step in the population.
ax = plt.subplot(3,1,3)
ax.plot(fr_exc)
plt.show()
