#   ANNarchy - NMDA
#
#   A single IF neuron with two non-linear synapses.
# 
#   This is a reimplementation of the PyNN example:
#
#   http://brian.readthedocs.org/en/latest/examples-synapses_nonlinear_synapses.html
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay

from ANNarchy import *

setup(dt=0.1)

# Neurons
Linear = Neuron(equations="dv/dt = 0.1", spike="v>1.0", reset="v=0.0")
Integrator = Neuron(equations="dv/dt = 0.1*(g_exc -v)", spike="v>2.0", reset="v=0.0")

# Non-linear synapse
NMDA = Synapse(
    parameters = """
    tau = 10.0 : projection
    """,
    equations = """
    tau * dx/dt = -x
    tau * dg/dt = -g +  x * (1 -g)
    """, 
    pre_spike = "x += w",
    psp = "g"
)

# Populations

input = Population(geometry=2, neuron=Linear)
input.v = [0.0, 0.5]
neurons = Population(geometry=1, neuron=Integrator)

# Projection
proj = Projection(pre=input, post=neurons, target='exc', synapse=NMDA).connect_from_matrix(weights=[[1.0, 10.0]])

# Compile the network
compile()

# Start recording
m = Monitor(neurons, 'v')
w = Monitor(proj[0], 'g')

# Simulate for 100 ms
simulate(100.0)

# Retrieve recordings
v = m.get('v')[:, 0]
s = w.get('g')

# Plot the recordings
import matplotlib.pyplot as plt
plt.subplot(2,1,1)
plt.plot(s[:, 0])
plt.plot(s[:, 1])
plt.subplot(2,1,2)
plt.plot(v)
plt.show()
