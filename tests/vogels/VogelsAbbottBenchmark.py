from ANNarchy import *
from time import time

dt = 0.1
setup(dt=dt)

# ###########################################
# Defining network model parameters
# ###########################################

NE = 3200        # Number of excitatory cells
NI = 800          # Number of inhibitory cells
duration = 1.0 * 1000.0 # Total time of the simulation

# ###########################################
# Neuron model
# ###########################################

LIF = SpikeNeuron(
    parameters="""
        Urest = -60.0  : population
        Uexc = 0.0  : population
        Uinh = -80.0  : population
        T = -50.0   : population
        I = 200.0 : population
        tau_m = 20.0   : population
        tau_exc = 5.0   : population
        tau_inh = 10.0  : population
    """,
    equations="""
        tau_m * dv/dt = (Urest - v) + g_exc * (Uexc - v) + g_inh * (Uinh - v ) + I

        tau_exc * dg_exc/dt = - g_exc 
        tau_inh * dg_inh/dt = - g_inh 
    """,
    spike = """
        v > T
    """,
    reset = """
        v = Urest
    """,
    refractory = 5.0
)


# ###########################################
# Initialize neuron group
# ###########################################
P = Population(geometry=NE+NI, neuron=LIF)
Pe = P[:NE]
Pi = P[NE:]

Input = PoissonPopulation(geometry=200, rates="if t < 50.0: 10.0 else: 0.0")

# ###########################################
# Connecting the network
# ###########################################

con_e = Projection(pre=Pe, post=P, target='exc').connect_fixed_probability(weights=0.4, probability=0.02, delays=0.8)
con_i = Projection(pre=Pi, post=P, target='inh').connect_fixed_probability(weights=5.1, probability=0.02, delays=0.8)
con_input = Projection(pre=Input, post=Pe, target='exc').connect_fixed_probability(weights=0.4, probability=0.01, delays=0.8)


# ###########################################
# Compile the network
# ###########################################
compile()

# ###########################################
# Run without plasticity
# ###########################################
P.start_record('spike')
simulate(duration, measure_time=True)
data = P.get_record()
P.stop_record()


# ###########################################
# Make plots
# ###########################################
spikes = raster_plot(data['spike'])

from pylab import *
plot(dt*spikes[:, 0], spikes[:, 1], '.', markersize=0.1)
title('Spikes')
show()
