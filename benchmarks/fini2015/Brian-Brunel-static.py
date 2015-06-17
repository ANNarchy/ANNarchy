from brian import *
import random

# ###########################################
# Configuration
# ###########################################
set_global_preferences(useweave=True)
set_global_preferences(usecodegen=True)
#set_global_preferences(openmp=True)

# ###########################################
# Network parameters
# ###########################################
NE = 10000 # Number of excitatory neurons
NI = 2500 # Number of excitatory neurons
Nrec = 1000 # Number of neurons to record
J_ex  = 0.1 * mV # excitatory weight
J_in  = -0.5 * mV  # inhibitory weight
p_rate = 20. * Hz # external Poisson rate
delay = 1.5 * ms # synaptic delay
tau_m = 20.0*ms

simtime = 0.1*second # Simulation time

# ###########################################
# Neuron model
# ###########################################
eqs_neurons='''
dv/dt = -v/tau_m : volt
'''

# ###########################################
# Populations
# ###########################################
P = NeuronGroup(NE+NI, model=eqs_neurons, threshold=20.*mV, reset=0.*mV, refractory=2.*ms)
Pe = P.subgroup(NE)
Pi = P.subgroup(NI)

poisson = PoissonGroup(NE, rates=p_rate)

# ###########################################
# Projections
# ###########################################
ee = Synapses(Pe, P, model='w:mV',pre='v+=w')
for post in xrange(NE+NI):
    ee[random.sample(xrange(NE), NE/10), post] = True
ee.w = J_ex
ee.delay = delay
ii = Synapses(Pi, P, model='w:mV',pre='v+=w')
for post in xrange(NE+NI):
    ii[random.sample(xrange(NI), NI/10), post] = True
ii.w = J_in
ii.delay = delay
noise = Synapses(poisson, P, model='w:mV',pre='v+=w')
for post in xrange(NE+NI):
    noise[random.sample(xrange(NE), NE/10), post] = True
noise.w = J_ex
noise.delay = delay

# ###########################################
# Setting up monitors
# ###########################################
sm = SpikeMonitor(P[:Nrec])

# ###########################################
# Simulation
# ###########################################
print 'Start simulation'
from time import time
ts = time()
run(simtime)
print 'Simulation took', time() - ts, 's'
print 'Mean firing rate:', sm.nspikes/float(Nrec)*1000./100., 'Hz'

# ###########################################
# Make plots
# ###########################################
subplot(111)
raster_plot(sm, ms=2.)
xlabel("")
ylim(0, Nrec)
show()