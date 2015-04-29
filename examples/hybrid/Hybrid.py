#
#   ANNarchy - Hybrid network
#
#   Simple example showing hybrid spike/rate-coded networks.
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay
#
from ANNarchy import *

# Frequency corresponding to a rate r of 1.0
max_freq = 100.0

# Rate-coded input neuron
neuron = Neuron(
    equations="""
    r = if t < 500: 
            0.0 
        else: 
            if t < 1000.0: 
                0.5 
            else: 
                if t < 1500: 
                    1.0 
                else: 
                    if t > 2000.0 : 
                        (1.0 + sin(2*pi*(t-2000)/1000 -pi/2))/2.0 
                    else: 0.0 : min=0.0 """)

# Rate-coded population for input
pop1 = Population(geometry=200, neuron=neuron)
# Spiking population
pop2 = Rate2SpikePopulation(population=pop1, scaling=max_freq)
# Rate-coded population for the backward convertion
pop3 = Spike2RatePopulation(population=pop2, mode='window', window=50.0, smooth=20.0)
pop4 = Spike2RatePopulation(population=pop2, mode='isi', cut=4.0, smooth=20.0)

compile()

# Monitors
m1 = Monitor(pop1, 'r')
m2 = Monitor(pop2, 'spike')
m3 = Monitor(pop3, 'r')
m4 = Monitor(pop4, 'r')

# Simulate
simulate(4000.0)

# Get recordings
data1 = m1.get()
data2 = m2.get()
data3 = m3.get()
data4 = m4.get()

# Raster plot of the spiking population
t, n = m2.raster_plot(data2['spike'])

# Spike times of a single neuron
singlespike = data2['spike'][0]

# Plot the results
import pylab as plt

ax1 = plt.subplot(3,1,1)
ax1.plot(t, n, '.', markersize=0.5)
ax1.set_title('a) Raster plot')
ax1.set_ylabel('# neurons')
ax1.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
ax1.set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(max_freq*data1['r'][:, 0], label='r')
ax2.plot(data4['r'][:,0], label='ISI')
ax2.plot(data3['r'][:,0], label='W')
ax2.plot(np.array(singlespike), 2.0*max_freq*np.ones(np.array(singlespike).shape), 'b.')
lg = ax2.legend(loc=2)
lg.draw_frame(False)
ax2.set_ylim((0.0, 2.2*max_freq))
ax2.set_title('b) Single neuron')
ax2.set_ylabel('Firing rate (Hz)')

ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(max_freq*data1['r'][:,0], label='r')
ax3.plot(np.mean(data4['r'], axis=1), label='ISI')
ax3.plot(np.mean(data3['r'], axis=1), label='W')
ax3.set_ylim((0.0, 1.2*max_freq))
lg = ax3.legend(loc=2)
lg.draw_frame(False)
ax3.set_title('c) Population firing rate')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Mean firing rate (Hz)')

plt.show()