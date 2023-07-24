from ANNarchy import *
setup(dt=0.1)

# ###########################################
# Neuron model
# ###########################################
COBA = Neuron(
    parameters="""
        El = -60.0          : population
        Vr = -60.0          : population
        Erev_exc = 0.0      : population
        Erev_inh = -80.0    : population
        Vt = -50.0          : population
        tau = 20.0          : population
        tau_exc = 5.0       : population
        tau_inh = 10.0      : population
        I = 20.0            : population
    """,
    equations="""
        tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I

        tau_exc * dg_exc/dt = - g_exc
        tau_inh * dg_inh/dt = - g_inh
    """,
    spike = "v > Vt",
    reset = "v = Vr",
    refractory = 5.0
)

# ###########################################
# Create population
# ###########################################
P = Population(geometry=4000, neuron=COBA)
Pe = P[:3200]
Pi = P[3200:]
P.v = Normal(-55.0, 5.0)
P.g_exc = Normal(4.0, 1.5)
P.g_inh = Normal(20.0, 12.0)

# ###########################################
# Connect the network
# ###########################################
Ce = Projection(pre=Pe, post=P, target='exc')
Ce.connect_fixed_probability(weights=0.6, probability=0.02)
Ci = Projection(pre=Pi, post=P, target='inh')
Ci.connect_fixed_probability(weights=6.7, probability=0.02)

compile()

# ###########################################
# Simulate
# ###########################################
m = Monitor(P, ['spike'])
simulate(1000.0, measure_time=True)
data = m.get('spike')

# ###########################################
# Make plots
# ###########################################
t, n = m.raster_plot(data)
print('Mean firing rate in the population: ' + str(len(t) / 4000.) + 'Hz')

import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, n, '.', markersize=0.5)
plt.xlabel('Time (ms)')
plt.ylabel('# neuron')

plt.figure()
plt.title("COBA statistics")
plt.subplots_adjust(hspace=0.3, wspace=0.3)

exc_isi = m.inter_spike_interval(data, ranks=Pe.ranks)
inh_isi = m.inter_spike_interval(data, ranks=Pi.ranks)

ax = plt.subplot(2,2,1)
ax.set_title("excitatory")
plt.hist(exc_isi, bins=10**np.linspace(0, 3, 33))
ax.set_xlabel("ISI [ms]")
ax.set_ylabel("n in bin")
ax.set_xscale("log")

ax2 = plt.subplot(2,2,2)
ax2.set_title("inhibitory")
ax2.hist(inh_isi, bins=10**np.linspace(0, 3, 33))
ax2.set_xlabel("ISI [ms]")
ax2.set_ylabel("n in bin")
ax2.set_xscale("log")

exc_isi_cv = m.coefficient_of_variation(data, ranks=Pe.ranks)
inh_isi_cv = m.coefficient_of_variation(data, ranks=Pi.ranks)

ax3 = plt.subplot(2,2,3)
plt.hist(exc_isi_cv, bins=15)
ax3.set_xlabel("ISI CV[ms]")
ax3.set_ylabel("n in bin")

ax4 = plt.subplot(2,2,4)
plt.hist(inh_isi_cv, bins=15)
ax4.set_xlabel("ISI CV[ms]")
ax4.set_ylabel("n in bin")

plt.show()
