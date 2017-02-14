#   ANNarchy - AEIF_cond_exp
#
#   Adaptive exponential integrate-and-fire model. 
#
#   http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
#
# Introduced in 
#
# Brette R. and Gerstner W. (2005), Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity, J. Neurophysiol. 94: 3637 - 3642.
# 
#   This is a reimplementation of the Brian example:
#
#   http://briansimulator.org/docs/examples-frompapers_Brette_Gerstner_2005.html
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay

from ANNarchy import *

# Set the discretization step
dt = 0.1
setup(dt=dt)

# Create a population with one AdEx neuron
pop = Population(geometry=1, neuron=EIF_cond_exp_isfa_ista)

# Regular spiking (paper)
pop.tau_w, pop.a, pop.b, pop.v_reset = 144.0,  4.0, 0.0805, -70.6

# Bursting 
#pop.tau_w, pop.a, pop.b, pop.v_reset = 20.0, 4.0, 0.5, pop.v_thresh + 5.0

# Fast spiking
#pop.tau_w, pop.a, pop.b, pop.v_reset = 144.0, 2000.0*pop.cm/144.0, 0.0, -70.6 


# Compile the network
compile()

# Start recording
m = Monitor(pop, ['spike', 'v', 'w'])

# Add current of 1 nA and simulate
simulate(20.0)
pop.i_offset = 1.0
simulate(100.0)
pop.i_offset = 0.0
simulate(20.0)

# Retrieve the results
data = m.get()
spikes = data['spike'][0]
v = data['v'][:, 0]
w = data['w'][:, 0]
if len(spikes)>0:
    v[spikes] = 20.0

# Plot the activity
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(dt*np.arange(140.0/dt), v)
plt.ylabel('v')
plt.title('Adaptive exponential integrate-and-fire')
plt.subplot(2,1,2)
plt.plot(dt*np.arange(140.0/dt), w)
plt.xlabel('Time (ms)')
plt.ylabel('w')
plt.show()
