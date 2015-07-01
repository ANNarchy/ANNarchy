from ANNarchy import *

# ###########################################
# Configuration
# ###########################################
setup(dt=0.1)

# ###########################################
# Network parameters
# ###########################################
NE = 8000        # Number of excitatory cells
NI = NE/4        # Number of inhibitory cells
duration = 10.0 * 1000.0 # Total time of the simulation
epsilon = 0.02

# ###########################################
# Neuron model
# ###########################################
COBA = Neuron(
    parameters="""
        El = -60.0  : population
        Vr = -60.0  : population
        Erev_inh = -80.0  : population
        Vt = -50.0   : population
        C = 200.0   : population
        gL = 10.0   : population
        tau_exc = 5.0   : population
        tau_inh = 10.0  : population
        I = 200.0 : population
    """,
    equations="""
        C * dv/dt = gL*(El - v) - g_exc * v + g_inh * (Erev_inh - v) + I

        tau_exc * dg_exc/dt = - g_exc 
        tau_inh * dg_inh/dt = - g_inh 
    """,
    spike = "v > Vt",
    reset = "v = Vr",
    refractory = 5.0
)

# ###########################################
# STDP synapse model
# ###########################################
STDP = Synapse(
    parameters="""
        tau_stdp = 20.0 : postsynaptic
        gmax = 100.0 : postsynaptic
        eta = 0.01 : postsynaptic
        alpha = 0.012 : postsynaptic
    """,
    equations = """
        tau_stdp * dApre/dt = -Apre : event-driven
        tau_stdp * dApost/dt = -Apost : event-driven
    """,
    pre_spike="""
        g_target += w
        Apre += 1.0
        w = clip(w + (Apost - alpha) * eta, 0.0, gmax)
    """,                  
    post_spike="""
        Apost += 1.0
        w = clip (w + Apre * eta, 0.0, gmax)
    """
)

# ###########################################
# Population
# ###########################################
P = Population(geometry=NE+NI, neuron=COBA)
Pe = P[:NE]
Pi = P[NE:]

# ###########################################
# Projection
# ###########################################
con_e = Projection(Pe, P, 'exc').connect_fixed_probability(weights=0.3, probability=epsilon)
con_ie = Projection(Pi, Pe ,'inh', STDP).connect_fixed_probability(weights=1e-10, probability=epsilon)
con_ii = Projection(Pi, Pi, 'inh').connect_fixed_probability(weights=3., probability=epsilon)

compile()

# ###########################################
# Simulation
# ###########################################
m = Monitor(P, 'spike')
simulate(duration , measure_time=True)
data = m.get()

# ###########################################
# Data analysis
# ###########################################
t, n = m.raster_plot(data['spike'])
print 'Number of spikes:', len(t)
print 'Mean firing rate:', len(t)/float(NE+NI)/duration*1000.0

from pylab import *
subplot(211)
plot(t, n, '.', markersize=0.5)
title("Before")
xlabel('')
ylabel('Neuron number')
xlim(0, 0.2*1e3)
subplot(212)
plot(t, n, '.', markersize=0.5)
title("After")
xlabel('Time (ms)')
ylabel('Neuron number')
xlim(9.8*1e3, 10*1e3)
show()
