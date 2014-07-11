from ANNarchy import *

setup(dt=0.1)

Input = SpikeSourceArray([[0.0, 50.0, 90.0], [55.0, 100.0]])

ifcell = Population(1, IF_cond_exp)
ifcell.set( {  'i_offset' : 0.1,    'tau_refrac' : 3.0,
                'v_thresh' : -51.0,  'tau_syn_E'  : 2.0,
                'tau_syn_I': 5.0,    'v_reset'    : -70.0,
                'e_rev_E'  : 0.,     'e_rev_I'    : -80.0 })


spike_sourceE = SpikeSourceArray(spike_times= [float(i) for i in range(5,105,10)] )
spike_sourceI = SpikeSourceArray(spike_times= [float(i) for i in range(155,255,10)])

connE = Projection(spike_sourceE, ifcell, 'exc').connect_all_to_all(weights=0.006, delays=2.0)
connI = Projection(spike_sourceI, ifcell, 'inh').connect_all_to_all(weights=0.02,  delays=4.0)

compile()

start_record({ifcell: ['spike', 'v']})

simulate(200.0)

data = get_record()

from pylab import *
plot(data[ifcell]['v']['data'][0])
show()


