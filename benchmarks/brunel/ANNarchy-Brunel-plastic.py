from ANNarchy import *
from time import time
from pylab import *

# ###########################################
# Configuration
# ###########################################
setup(dt=0.1)
record = True
plot_all = True

# ###########################################
# Parameters
# ###########################################
g       = 5.0    # Ratio of IPSP to EPSP amplitude: J_I/J_E
delay   = 1.5    # synaptic delay in ms
V_th    = 20.0   # Spike threshold in mV
simtime = 300.0   # how long shall we simulate [ms]
Nrec    = 50     # Number of neurons to record 

NE = 8000
NI = 2000
N  = NE+NI
CE = int(NE/10) # number of excitatory synapses per neuron
CI = int(NI/10) # number of inhibitory synapses per neuron
JE = 0.1
JI = -g*JE

# ###########################################
# Neuron model
# ###########################################
IAF = Neuron(
    parameters="""
        v_th = 20.0 : population
        tau_m = 20.0 : population
    """,
    equations="""
        # The real equation should be:
        # tau_m * dv/dt = -v + g_exc + g_inh
        # But incoming spikes increment the membrane potential directly
        # So we apply the numerical method by hand
        v += g_exc + g_inh - dt*v/tau_m
    """,
    spike="v > v_th",
    reset="v = 0.0",
    refractory=2.0
)

# ###########################################
# Synapse model
# ###########################################
STDP = Synapse(
    parameters="""
        tau_stdp = 20.0 : postsynaptic
        wmax = 0.3 : postsynaptic
        lbda = 0.01 : postsynaptic
        alpha = 2.02 : postsynaptic
    """,
    equations = """
        tau_stdp * dApre/dt = -Apre : event-driven
        tau_stdp * dApost/dt = -Apost : event-driven
    """,
    pre_spike="""
        g_target += w
        Apre += 1.0
        w -= lbda * alpha * w * Apost : min=0.0
    """,                  
    post_spike="""
        Apost += 1.0
        w += lbda * (wmax - w) * Apre : max=wmax 
    """
)

# ###########################################
# Populations
# ###########################################
P = Population(geometry=N, neuron=IAF)
PE= P[:NE]
PI= P[NE:]
P.v = Uniform(-V_th, 0.95*V_th)

noise = PoissonPopulation(geometry=10000, rates=20.0)

# ###########################################
# Projections
# ###########################################
ee = Projection(PE, PE, 'exc', STDP)
ee.connect_fixed_number_pre(number=CE, weights=Uniform(0.5*JE, 1.5*JE), delays=delay)
ei = Projection(PE, PI, 'exc')
ei.connect_fixed_number_pre(number=CE, weights=Uniform(0.5*JE, 1.5*JE), delays=delay)
ii = Projection(PI, P , 'inh')
ii.connect_fixed_number_pre(number=CI, weights=JI, delays=delay)
noisy = Projection(noise, P, 'exc')
noisy.connect_fixed_number_pre(number=1000, weights=JE, delays=delay)

compile()

# ###########################################
# Simulation
# ###########################################
print 'Start simulation'
if record:
    m = Monitor(P[:Nrec], 'spike')
simulate(simtime, measure_time=True)
if record:
    data = m.get()

# ###########################################
# Data analysis
# ###########################################
if record:
    t, n = m.raster_plot(data['spike'])
    print 'Mean firing rate:', len(t)/float(Nrec)*1000./simtime, 'Hz'
    if plot_all:
        plot(t, n, '.')
        show()

# Histogram of the weights after learning
if plot_all:
    weights=[]
    for i in xrange(Nrec):
        ws = ee.dendrite(i).w
        for w in ws:
            weights.append(w)
    hist(weights, bins=100)
    xlabel('Synaptic weight [pA]')
    show()