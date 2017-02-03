"""
Network (CUBA) with short-term synaptic plasticity for excitatory synapses (Depressing at long timescales, facilitating at short timescales).
Adapetd from :
http://www.briansimulator.org/docs/examples-plasticity_short_term_plasticity2.html
"""
from ANNarchy import *

duration = 1000.0
setup(dt=0.1)

# ###########################################
# Define the neurons
# ###########################################
LIF = Neuron(
    parameters = """
    tau_m = 20.0 : population
    tau_e = 5.0 : population
    tau_i = 10.0 : population
    E_rest = -49.0 : population
    E_thresh = -50.0 : population
    E_reset = -60.0 : population
    """,
    equations = """
    tau_m * dv/dt = E_rest -v + g_exc - g_inh 
    tau_e * dg_exc/dt = -g_exc 
    tau_i * dg_inh/dt = -g_inh 
    """,
    spike = "v > E_thresh",
    reset = "v = E_reset"
)


# ###########################################
# Define the synapse
# ###########################################
STP = Synapse(
    parameters = """
    tau_rec = 200.0 : projection
    tau_facil = 20.0 : projection
    U = 0.2 : projection
    """,
    equations = """
    dx/dt = (1 - x)/tau_rec : init = 1.0, event-driven
    du/dt = (U - u)/tau_facil : init = 0.2, event-driven   
    """,
    pre_spike="""
    g_target += w * u * x
    x *= (1 - u)
    u += U * (1 - u)
    """
)

# ###########################################
# Create the populations
# ###########################################
P = Population(geometry=4000, neuron=LIF)
P.v = Uniform(-60.0, -50.0)
Pe = P[:3200]
Pi = P[3200:]

# ###########################################
# Create the projections
# ###########################################
con_e = Projection(pre=Pe, post=P, target='exc', synapse = STP).connect_fixed_probability(weights=1.62, probability=0.02)
con_i = Projection(pre=Pi, post=P, target='inh').connect_fixed_probability(weights=9.0, probability=0.02)

# ###########################################
# Compile the network
# ###########################################
compile()

# ###########################################
# Run without plasticity
# ###########################################
m = Monitor(P, 'spike')
simulate(duration, measure_time=True)
data = m.get()

# ###########################################
# Make plots
# ###########################################
t, n = m.raster_plot(data['spike'])
rates = m.population_rate(data['spike'], 5.0)
print('Total number of spikes: ' + str(len(t)))

import matplotlib.pyplot as plt
plt.subplot(211)
plt.plot(t, n, '.')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron number')
plt.subplot(212)
plt.plot(np.arange(rates.size)*dt(), rates)
plt.show()
