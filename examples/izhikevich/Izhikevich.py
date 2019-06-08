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

# Create the excitatory and inhibitory population
pop = Population(geometry=1000, neuron=Izhikevich)
Exc = pop[:800]                 ; Inh = pop[800:]

# Set the population parameters
re = np.random.random(800)      ; ri = np.random.random(200)
Exc.noise = 5.0                 ; Inh.noise = 2.0
Exc.a = 0.02                    ; Inh.a = 0.02 + 0.08 * ri
Exc.b = 0.2                     ; Inh.b = 0.25 - 0.05 * ri
Exc.c = -65.0 + 15.0 * re**2    ; Inh.c = -65.0
Exc.d = 8.0 - 6.0 * re**2       ; Inh.d = 2.0
Exc.v = -65.0                   ; Inh.v = -65.0
Exc.u = Exc.v * Exc.b           ; Inh.u = Inh.v * Inh.b

# Create the projections
exc_proj = Projection(pre=Exc, post=pop, target='exc')
exc_proj.connect_all_to_all(weights=Uniform(0.0, 0.5))

inh_proj = Projection(pre=Inh, post=pop, target='inh')
inh_proj.connect_all_to_all(weights=Uniform(0.0, 1.0))

# Compile
compile()

# Start recording the spikes in the network to produce the plots
M = Monitor(pop, ['spike', 'v'])

# Simulate 1 second
simulate(1000.0, measure_time=True)

# Retrieve the spike recordings and the membrane potential
spikes = M.get('spike')
v = M.get('v')

# Compute the raster plot
t, n = M.raster_plot(spikes)

# Compute the population firing rate
fr = M.histogram(spikes)

# Plot the results
import matplotlib.pyplot as plt
# First plot: raster plot
ax = plt.subplot(3,1,1)
ax.plot(t, n, 'b.', markersize=1.0)
# Second plot: membrane potential of a single excitatory cell
ax = plt.subplot(3,1,2)
ax.plot(v[:, 15]) # for example
# Third plot: number of spikes per step in the population.
ax = plt.subplot(3,1,3)
ax.plot(fr)
plt.show()
