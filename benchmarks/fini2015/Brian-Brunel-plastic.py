from brian import *
import random

# ###########################################
# Configuration
# ###########################################
set_global_preferences(useweave=True)
set_global_preferences(usecodegen=True)
#set_global_preferences(openmp=True)

record = True
plot_all = True

# ###########################################
# Network parameters
# ###########################################
NE = 8000 # Number of excitatory neurons
NI = 2000 # Number of excitatory neurons
Nrec = 50 # Number of neurons to record
J_ex  = 0.1 # excitatory weight
J_in  = -0.5  # inhibitory weight
p_rate = 20. * Hz # external Poisson rate
delay = 1.5 * ms # synaptic delay
tau_m = 20.0 * ms

taupre = 20.*ms
taupost = 20.*ms
wmax = 0.3 
lbda = 0.01 
alpha = 2.02 

simtime = 0.3*second # Simulation time

# ###########################################
# Neuron model
# ###########################################
eqs_neurons='''
dv/dt = -v/tau_m : 1
'''

# ###########################################
# Synapse model
# ###########################################
model='''w:1
         dApre/dt=-Apre/taupre : 1 (event-driven)
         dApost/dt=-Apost/taupost : 1 (event-driven)'''
pre_spike = '''v+=w
         Apre += 1
         w-= lbda * alpha * w * Apost 
'''
post_spike = '''
         Apost += 1
         w += lbda * (wmax - w) * Apre
'''

# ###########################################
# Population
# ###########################################
P = NeuronGroup(NE+NI, model=eqs_neurons, threshold=20., reset=0., refractory=2.*ms)
Pe = P.subgroup(NE)
Pi = P.subgroup(NI)
P.v = (-20. + 1.95*20.*rand(NE+NI))

poisson = PoissonGroup(NE+NI, rates=p_rate)

# ###########################################
# Projections
# ###########################################
# EE
ee = Synapses(Pe, Pe, model=model,pre=pre_spike, post=post_spike)
for post in xrange(NE):
    ee[random.sample(xrange(NE), NE/10), post] = True
ee.w[:,:]='(0.5+rand())*J_ex'
ee.delay = delay
# EI
ei = Synapses(Pe, Pi, model='w:1',pre='v+=w')
for post in xrange(NI):
    ei[random.sample(xrange(NE), NE/10), post] = True
ei.w[:,:]='(0.5+rand())*J_ex'
ei.delay = delay
# II
ii = Synapses(Pi, P, model='w:1',pre='v+=w')
for post in xrange(NE+NI):
    ii[random.sample(xrange(NI), NI/10), post] = True
ii.w = J_in
ii.delay = delay
noise = Synapses(poisson, P, model='w:1',pre='v+=w')
for post in xrange(NE+NI):
    noise[random.sample(xrange(NE+NI), (NE+NI)/10), post] = True
noise.w = J_ex
noise.delay = delay

# ###########################################
# Setting up monitors
# ###########################################
if record:
    sm = SpikeMonitor(P[:Nrec])

# ###########################################
# Simulation
# ###########################################
print 'Start simulation'
from time import time
ts = time()
run(simtime)
print 'Simulation took', time() - ts, 's'
if record:
    print 'Mean firing rate:', sm.nspikes/float(Nrec)/float(simtime), 'Hz'

# ###########################################
# Make plots
# ###########################################
if plot_all:
    if record:
        raster_plot(sm, ms=2.)
        show()
    # Weights
    weights = ee.w[:, :Nrec].flatten()
    hist(weights, bins=100)
    xlabel('Synaptic weight [pA]')
    show()
