#   ANNarchy - IF_cond_exp
#
#   A single IF neuron with exponential, conductance-based synapses, fed by two spike sources.
# 
#   This is a reimplementation of the PyNN example:
#
#   http://www.neuralensemble.org/trac/PyNN/wiki/Examples/IF_cond_exp
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay

from ANNarchy import *

# Parameters
dt = 0.1
tstop = 200.0

# Setup
setup(dt=dt)

# Input populations with predetermined spike times
spike_sourceE = SpikeSourceArray(spike_times= [float(i) for i in range(5,105,10)] )
spike_sourceI = SpikeSourceArray(spike_times= [float(i) for i in range(155,255,10)])

# Population with one IF_cond_exp neuron
ifcell = Population(1, IF_cond_exp)
ifcell.set( 
    {   'i_offset' : 0.1,    'tau_refrac' : 3.0,
        'v_thresh' : -51.0,  'tau_syn_E'  : 2.0,
        'tau_syn_I': 5.0,    'v_reset'    : -70.0,
        'e_rev_E'  : 0.,     'e_rev_I'    : -80.0 } )


# Projections
connE = Projection(spike_sourceE, ifcell, 'exc').connect_all_to_all(weights=0.006, delays=2.0)
connI = Projection(spike_sourceI, ifcell, 'inh').connect_all_to_all(weights=0.02,  delays=4.0)

# Compile the network
compile()

# Simulate
m = Monitor(ifcell, ['spike', 'v'])
simulate(tstop)
data = m.get()

# Show the result
import matplotlib.pyplot as plt
plt.plot(dt*np.arange(tstop/dt), data['v'][:, 0])
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.ylim([-66.0, -61.0])
plt.title('IF_cond_exp')
plt.show()

