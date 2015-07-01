from brian import *
import numpy

# ###########################################
# Configuration
# ###########################################
set_global_preferences(useweave=True)
set_global_preferences(usecodegen=True)
set_global_preferences(openmp=True)

# ###########################################
# Network parameters
# ###########################################
NE = 8000          # Number of excitatory cells
NI = NE/4          # Number of inhibitory cells

w = 1.*nS           # Basic weight unit
tau_ampa = 5.0*ms   # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms  # GABAergic synaptic time constant
epsilon = 0.02      # Sparseness of synaptic connections

eta = 1e-2          # Learning rate
tau_stdp = 20*ms    # STDP time constant

simtime = 10*second # Simulation time

# ###########################################
# Neuron model
# ###########################################
gl = 10.0*nsiemens   # Leak conductance
el = -60*mV          # Resting potential
er = -80*mV          # Inhibitory reversal potential
vt = -50.*mV         # Spiking threshold
memc = 200.0*pfarad  # Membrane capacitance
bgcurrent = 200*pA   # External current

eqs_neurons='''
dv/dt=(-gl*(v-el)-(g_ampa*w*v+g_gaba*(v-er)*w)+bgcurrent)/memc : volt
dg_ampa/dt = -g_ampa/tau_ampa : 1
dg_gaba/dt = -g_gaba/tau_gaba : 1
'''

# ###########################################
# Population
# ###########################################
neurons=NeuronGroup(NE+NI, model=eqs_neurons, threshold=vt, reset=el, refractory=5*ms)
Pe=neurons.subgroup(NE)
Pi=neurons.subgroup(NI)

# ###########################################
# Projections
# ###########################################
con_e = Connection(Pe, neurons, 'g_ampa', weight=0.3, sparseness=epsilon)
con_ie = Connection(Pi, Pe ,'g_gaba', weight=1e-10, sparseness=epsilon)
con_ii = Connection(Pi, Pi, 'g_gaba', weight=3, sparseness=epsilon)

# ###########################################
# Inhibitory Plasticity
# ###########################################
alpha = 3*Hz*tau_stdp*2  # Target rate parameter
gmax = 100               # Maximum inhibitory weight

eqs_stdp_inhib = '''
dA_pre/dt=-A_pre/tau_stdp : 1
dA_post/dt=-A_post/tau_stdp : 1
'''

stdp_ie = STDP(
    con_ie, 
    eqs=eqs_stdp_inhib, 
    pre='A_pre+=1.; w+=(A_post-alpha)*eta;',
    post='A_post+=1.; w+=A_pre*eta;', 
    wmax=gmax
)

# ###########################################
# Setting up monitors
# ###########################################
sm = SpikeMonitor(neurons)

# ###########################################
# Simulation
# ###########################################
run(simtime, report='text')
print 'Total spikes:', sm.nspikes
print 'Mean firing rate:', (sm.nspikes)/float(NE+NI)/float(simtime)

# ###########################################
# Make plots
# ###########################################
subplot(211)
raster_plot(sm,ms=1.)
title("Before")
xlabel("")
xlim(0, 0.2*1e3)
subplot(212)
raster_plot(sm,ms=1.)
title("After")
xlim((simtime-0.2*second)*1e3, simtime*1e3)
show()