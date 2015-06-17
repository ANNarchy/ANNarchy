#
#   ANNarchy - Hybrid network
#
#   Simple example showing hybrid spike/rate-coded networks. 
#   Reproduces Fig.4 of (Vitay, Dinkelbach and Hamker, 2015)
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay
#
from ANNarchy import *

setup(dt=0.1)

# Rate-coded input neuron
input_neuron = Neuron(
    parameters = "baseline = 0.0",
    equations = "r = baseline"
)
# Rate-coded output neuron
simple_neuron = Neuron(
    equations = "r = sum(exc)"
)

# Rate-coded population for input
pop1 = Population(geometry=1, neuron=input_neuron)

# Poisson Population to encode
pop2 = PoissonPopulation(geometry=1000, target="exc")
proj = Projection(pop1, pop2, 'exc').connect_all_to_all(weights=1.)

# Rate-coded population to decode
pop3 = Population(geometry=1000, neuron =simple_neuron)
proj = DecodingProjection(pop2, pop3, 'exc', window=10.0)
def diagonal(pre, post, weights):
    "Simple connector pattern to progressively connect each post-synaptic neuron to a growing number of pre-synaptic neurons"
    csr = CSR()
    for rk_post in range(post.size):
        csr.add(rk_post, range((rk_post+1)), [weights], [0] )
    return csr
proj.connect_with_func(method=diagonal, weights=1.)

compile()

# Monitors
m1 = Monitor(pop1, 'r')
m2 = Monitor(pop2, 'spike')
m3 = Monitor(pop3, 'r')

# Simulate
duration = 250.
# 0 Hz
pop1.baseline = 0.0
simulate(duration)
# 10 Hz
pop1.baseline = 10.0
simulate(duration)
# 50 Hz
pop1.baseline = 50.0
simulate(duration)
# 100 Hz
pop1.baseline = 100.0
simulate(duration)

# Get recordings
data1 = m1.get()
data2 = m2.get()
data3 = m3.get()

# Raster plot of the spiking population
t, n = m2.raster_plot(data2['spike'])

# Variance of the the decoded firing rate
data_10 = data3['r'][1.0*duration/dt():2*duration/dt(), :]
data_50 = data3['r'][2.0*duration/dt():3*duration/dt(), :]
data_100 = data3['r'][3.0*duration/dt():4*duration/dt(), :]
var_10 = np.mean(np.abs((data_10 - 10.)/10.), axis=0)
var_50 = np.mean(np.abs((data_50 - 50.)/50.), axis=0)
var_100 = np.mean(np.abs((data_100 - 100.)/100.), axis=0)

### Plot the results
from pylab import *
subplot(3,1,1)
plot(t, n, '.', markersize=0.5)
title('a) Raster plot')
xlabel('Time (ms)')
ylabel('# neurons')
xlim((0, 4*duration))

subplot(3,1,2)
plot(np.arange(0, 4*duration, 0.1), data1['r'][:, 0], label='Original firing rate')
plot(np.arange(0, 4*duration, 0.1), data3['r'][:, 999], label='Decoded firing rate')
legend(frameon=False, loc=2)
title('b) Decoded firing rate')
xlabel('Time (ms)')
ylabel('Activity (Hz)')

subplot(3,1,3)
plot(var_10, label='10 Hz')
plot(var_50, label='50 Hz')
plot(var_100, label='100 Hz')
legend(frameon=False)
title('c) Precision')
xlabel('# neurons used for decoding')
ylabel('Normalized error')
ylim((0,1))

show()
