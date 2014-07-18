from ANNarchy import *

dt=1.0
setup(dt = dt)

pop1 = SpikeSourceArray(spike_times=[5.0])
pop2 = Population(geometry=1, neuron=IF_curr_exp)
proj = Projection(pop1, pop2, 'exc').connect_all_to_all(weights=1.0, delays=1.1)

compile()

pop2.start_record(['v'])

simulate(10.0)

data = pop2.get_record()
v = data['v']['data'][0]

from pylab import *
plot(dt*np.arange(len(v)), v)
show()