from __future__ import print_function
from ANNarchy import *
import random
from time import time

# Parameters
size = (32, 32) # input size
freq = 1.2 # nb_cycles/half-image
nb_stim = 40 # Number of grating per epoch
nb_epochs = 20 # Number of epochs
max_freq = 28. # Max frequency of the poisson neurons 

# Izhikevich RS neuron
RSNeuron = Neuron(
    parameters = """
        a = 0.02
        b = 0.2
        c = -65.
        d = 8.
        tau_ampa = 5.
        tau_nmda = 150.
        tau_gabaa = 6.
        tau_gabab = 150.
        vrev_ampa = 0.0
        vrev_nmda = 0.0
        vrev_gabaa = -70.0
        vrev_gabab = -90.0
    """ ,
    equations="""
        # Inputs
        I = g_ampa * (vrev_ampa - v) + g_nmda * nmda(v, -80.0, 60.0) * (vrev_nmda -v) + g_gabaa * (vrev_gabaa - v) + g_gabab * (vrev_gabab -v)
        # Midpoint scheme      
        dv/dt = (0.04 * v + 5.0) * v + 140.0 - u + I : init=-65., midpoint
        du/dt = a * (b*v - u) : init=-13., midpoint
        # Conductances
        tau_ampa * dg_ampa/dt = -g_ampa : exponential
        tau_nmda * dg_nmda/dt = -g_nmda : exponential
        tau_gabaa * dg_gabaa/dt = -g_gabaa : exponential
        tau_gabab * dg_gabab/dt = -g_gabab : exponential
    """ , 
    spike = """
        v >= 30.
    """, 
    reset = """
        v = c
        u += d
    """,
    functions = """
        nmda(v, t, s) = ((v-t)/(s))^2 / (1.0 + ((v-t)/(s))^2)
    """,
    refractory=1.0
)

# Nearest Neighbour STDP
nearest_neighbour_stdp = Synapse(
    parameters="""
        tau_plus  = 60. : postsynaptic
        tau_minus = 90. : postsynaptic
        A_plus  = 0.000045 : postsynaptic
        A_minus = 0.00003 : postsynaptic
        w_min = 0.0 : postsynaptic
        w_max = 0.03 : postsynaptic
    """,
    equations = """
        # Traces
        tau_plus  * dltp/dt = -ltp
        tau_minus * dltd/dt = -ltd
        # Nearest-neighbour
        w += if t_post >= t_pre: ltp else: - ltd : min=w_min, max=w_max
    """,
    pre_spike="""
        g_target += w
        ltp = A_plus 
    """,         
    post_spike="""
        ltd = A_minus 
    """
)

# STDP with homeostatic regulation
homeo_stdp = Synapse(
    parameters="""
        # STDP
        tau_plus  = 60. : postsynaptic
        tau_minus = 90. : postsynaptic
        A_plus  = 0.000045 : postsynaptic
        A_minus = 0.00003 : postsynaptic
        w_min = 0.0 : postsynaptic
        w_max = 0.03 : postsynaptic

        # Homeostatic regulation
        alpha = 0.1 : postsynaptic
        beta = 1.0 : postsynaptic
        gamma = 50. : postsynaptic
        Rtarget = 10. : postsynaptic
        T = 10000. : postsynaptic
    """,
    equations = """
        # Traces
        tau_plus  * dltp/dt = -ltp
        tau_minus * dltd/dt = -ltd  
        # Homeostatic values
        R = post.r : postsynaptic
        K = R/(T*(1.+fabs(1. - R/Rtarget) * gamma)) : postsynaptic
        # Nearest-neighbour
        stdp = if t_post >= t_pre: ltp else: - ltd 
        w += (alpha * w * (1- R/Rtarget) + beta * stdp ) * K : min=w_min, max=w_max
    """,
    pre_spike="""
        g_target += w
        ltp = A_plus 
    """,         
    post_spike="""
        ltd = A_minus 
    """ 
)

# Input population
OnPoiss = PoissonPopulation(size, rates=1.0)
OffPoiss = PoissonPopulation(size, rates=1.0)

# RS neuron for the input buffers
OnBuffer = Population(size, RSNeuron)
OffBuffer = Population(size, RSNeuron)

# Connect the buffers
OnPoissBuffer = Projection(OnPoiss, OnBuffer, ['ampa', 'nmda'])
OnPoissBuffer.connect_one_to_one(Uniform(0.2, 0.6))
OffPoissBuffer = Projection(OffPoiss, OffBuffer, ['ampa', 'nmda'])
OffPoissBuffer.connect_one_to_one(Uniform(0.2, 0.6))

# Excitatory and inhibitory neurons
Exc = Population(4, RSNeuron)
Inh = Population(4, RSNeuron)
Exc.compute_firing_rate(10000.)
Inh.compute_firing_rate(10000.)

# Input connections
OnBufferExc = Projection(OnBuffer, Exc, ['ampa', 'nmda'], homeo_stdp)
OnBufferExc.connect_all_to_all(Uniform(0.004, 0.015))
OffBufferExc = Projection(OffBuffer, Exc, ['ampa', 'nmda'], homeo_stdp)
OffBufferExc.connect_all_to_all(Uniform(0.004, 0.015))

# Competition
ExcInh = Projection(Exc, Inh, ['ampa', 'nmda'], homeo_stdp)
ExcInh.connect_all_to_all(Uniform(0.116, 0.403))
ExcInh.Rtarget = 75.
ExcInh.tau_plus = 51.
ExcInh.tau_minus = 78.
ExcInh.A_plus = -4.1e-5
ExcInh.A_minus = -1.5e-5
ExcInh.w_max = 1.0

InhExc = Projection(Inh, Exc, ['gabaa', 'gabab'])
InhExc.connect_all_to_all(Uniform(0.065, 0.259))

compile()

# Inputs
def get_grating(theta):
    x = np.linspace(-1., 1., size[0])
    y = np.linspace(-1., 1., size[1])
    xx, yy = np.meshgrid(x, y)
    z = np.sin(2.*np.pi*(np.cos(theta)*xx + np.sin(theta)*yy)*freq)
    return np.maximum(z, 0.), -np.minimum(z, 0.0)

# Initial weights
w_on_start = OnBufferExc.w
w_off_start = OffBufferExc.w

# Monitors
m = Monitor(Exc, 'r')

# Learning procedure
tstart = time()
for epoch in range(nb_epochs):
    stim_order = list(range(nb_stim))
    random.shuffle(stim_order)
    for stim in stim_order:
        # Generate a grating randomly
        rates_on, rates_off = get_grating(np.pi*stim/float(nb_stim))
        # Set it as input to the poisson neurons
        OnPoiss.rates = max_freq * rates_on
        OffPoiss.rates = max_freq * rates_off
        # Simulate for 2s
        simulate(2000.)
        # Relax the Poisson inputs
        OnPoiss.rates = 1.
        OffPoiss.rates = 1.
        # Simulate for 500ms
        simulate(500.)
    print('Epoch', epoch+1, 'done.')
print('Done in ', time()-tstart)

# Recordings
data = m.get('r')

# Final weights
w_on_end = OnBufferExc.w
w_off_end = OffBufferExc.w

# Plot
from pylab import *
subplot(241)
imshow((np.array(w_on_start[0])-np.array(w_off_start[0])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(242)
imshow((np.array(w_on_start[1])-np.array(w_off_start[1])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(243)
imshow((np.array(w_on_start[2])-np.array(w_off_start[2])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(244)
imshow((np.array(w_on_start[3])-np.array(w_off_start[3])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(245)
imshow((np.array(w_on_end[0])-np.array(w_off_end[0])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(246)
imshow((np.array(w_on_end[1])-np.array(w_off_end[1])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(247)
imshow((np.array(w_on_end[2])-np.array(w_off_end[2])).reshape((32,32)), aspect='auto', cmap='hot')
subplot(248)
imshow((np.array(w_on_end[3])-np.array(w_off_end[3])).reshape((32,32)), aspect='auto', cmap='hot')
show()

plot(data[:, 0])
show()