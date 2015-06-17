from brian import *
import time, cPickle
import numpy

# ###########################################
# Configuration
# ###########################################
numpy.random.seed(98765)
set_global_preferences(useweave=True)
set_global_preferences(usecodegen=True)
set_global_preferences(openmp=True)

# ###########################################
# Network parameters
# ###########################################
duration = 10. * second
# Time constants
taum = 20 * msecond
taue = 5 * msecond
taui = 10 * msecond
# Reversal potentials
Ee = 0. * mvolt
Ei = -80. * mvolt
El = - 60.0 * mvolt
I = 20. * mvolt

# ###########################################
# Neuron model
# ###########################################
eqs = Equations('''
dv/dt = (El-v + ge*(Ee-v) + gi*(Ei-v) + I) * (1./taum) : volt
dge/dt = -ge*(1./taue) : 1
dgi/dt = -gi*(1./taui) : 1
''')

# ###########################################
# Population
# ###########################################
P = NeuronGroup(4000, model=eqs, threshold=-50 * mvolt, \
              reset=-60 * mvolt, refractory=5 * msecond, 
              order=1, compile=True)
Pe = P.subgroup(3200)
Pi = P.subgroup(800)
P.v = (randn(len(P)) * 5. - 55.) * mvolt

# ###########################################
# Projections
# ###########################################
we = 6. / 10. # excitatory synaptic weight (voltage)
wi = 67. / 10. # inhibitory synaptic weight
mate = cPickle.load(open('exc.data', 'r'))
mati = cPickle.load(open('inh.data', 'r'))

Ce = Connection(Pe, P, 'ge')
#Ce.connect_random(weight=we, p=0.02, seed=seed)
Ce.connect_from_sparse(mate*we, column_access=False)
Ci = Connection(Pi, P, 'gi')
#Ci.connect_random(weight=wi, p=0.02, seed=seed)
Ci.connect_from_sparse(mati*wi, column_access=False)

# ###########################################
# Simulation
# ###########################################
# Record the number of spikes
Me = PopulationSpikeCounter(Pe)
Mi = PopulationSpikeCounter(Pi)
start_time = time.time()
run(duration)
duration = time.time() - start_time
print "Simulation time:", duration, "seconds"

# ###########################################
# Data analysis
# ###########################################
print Me.nspikes, "excitatory spikes"
print Mi.nspikes, "inhibitory spikes"
print Me.nspikes + Mi.nspikes, "total spikes"

