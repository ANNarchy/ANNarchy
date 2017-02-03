from ANNarchy import *
setup(dt=0.1)

# ###########################################
# Neuron model
# ###########################################
CUBA = Neuron(
    parameters="""
        El = -49.0      : population
        Vr = -60.0      : population
        Vt = -50.0      : population
        tau_m = 20.0    : population
        tau_exc = 5.0   : population
        tau_inh = 10.0  : population
    """,
    equations="""
        tau_m * dv/dt = (El - v) + g_exc + g_inh 

        tau_exc * dg_exc/dt = - g_exc 
        tau_inh * dg_inh/dt = - g_inh 
    """,
    spike = "v > Vt",
    reset = "v = Vr",
    refractory = 5.0
)


# ###########################################
# Create Population
# ###########################################
P = Population(geometry=4000, neuron=CUBA)
Pe = P[:3200]
Pi = P[3200:]
P.v = Uniform(-60.0, -50.0)

# ###########################################
# Connect the network
# ###########################################
we = 0.27 * 60.0 / 10.0 # 0.7 * (Vmean - E_rev_exc) / gL (mV)
wi = - 4.5 * 20.0 / 10.0 # 4.5 * (Vmean - E_rev_inh) / gL (mV)
Ce = Projection(pre=Pe, post=P, target='exc')
Ce.connect_fixed_probability(weights=we, probability=0.02)
Ci = Projection(pre=Pi, post=P, target='inh')
Ci.connect_fixed_probability(weights=wi, probability=0.02)

compile()

# ###########################################
# Simulate
# ###########################################
m = Monitor(P, ['spike'])
simulate(1000.0, measure_time=True)
data = m.get('spike')

###########################################
# Make plots
###########################################
t, n = m.raster_plot(data)
print('Mean firing rate in the population: ' + str(len(t) / 4000.) + 'Hz')

import matplotlib.pyplot as plt
plt.plot(t, n, '.')
plt.xlabel('Time (ms)')
plt.ylabel('# neuron')
plt.show()