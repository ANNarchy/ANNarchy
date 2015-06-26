from brian2 import *
import time, cPickle

# ###########################################
# Configuration
# ###########################################
numpy.random.seed(98765)
set_device('cpp_standalone')
prefs.devices.cpp_standalone.openmp_threads = 1

# ###########################################
# Network parameters
# ###########################################
taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -60*mV
Erev_exc = 0.*mV
Erev_inh = -80.*mV
I = 20. * mvolt

# ###########################################
# Neuron model
# ###########################################
eqs = '''
dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
dge/dt = -ge/taue : 1 
dgi/dt = -gi/taui : 1 
'''

# ###########################################
# Population
# ###########################################
P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms, method='euler')
P.v = (randn(len(P)) * 5. - 55.) * mvolt

# ###########################################
# Projections
# ###########################################
we = 0.6 # excitatory synaptic weight (voltage)
wi = 6.7 # inhibitory synaptic weight
Ce = Synapses(P, P, pre='ge += we')
Ci = Synapses(P, P, pre='gi += wi')
#Ce.connect('i<3200', p=0.02)
#Ci.connect('i>=3200', p=0.02)

# mate = cPickle.load(open('exc.data', 'r'))
# mati = cPickle.load(open('inh.data', 'r'))
Ce_pre, Ce_post = cPickle.load(open('exc.data', 'r')).nonzero()
Ci_pre, Ci_post = cPickle.load(open('inh.data', 'r')).nonzero()
Ce.connect(Ce_pre, Ce_post)
Ci.connect(Ci_pre+3200, Ci_post)



# ###########################################
# Simulation
# ###########################################
s_mon = SpikeMonitor(P)
# Run for 0 second in order to measure compilation time
t1 = time.time()
run(10. * second)
device.build(directory='COBA', compile=True, run=True, debug=False)
t2 = time.time()
print 'Done in', t2 - t1

# ###########################################
# Data analysis
# ###########################################
plot(s_mon.t/ms, s_mon.i, '.')
xlabel('Time (ms)')
ylabel('Neuron index')
show()
