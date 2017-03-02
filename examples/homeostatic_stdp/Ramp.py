"""
Homeostatic STDP mechanism.

Reimplementation of the Ramp model published in:

Carlson, K.D.; Richert, M.; Dutt, N.; Krichmar, J.L., "Biologically plausible models of homeostasis and STDP: Stability and learning in spiking neural networks," in Neural Networks (IJCNN), The 2013 International Joint Conference on , vol., no., pp.1-8, 4-9 Aug. 2013. doi: 10.1109/IJCNN.2013.6706961

Based on the Carlsim tutorial:

http://www.socsci.uci.edu/~jkrichma/CARLsim/doc/tut3_plasticity.html

"""
from __future__ import print_function
from ANNarchy import *

# Izhikevich RS neuron
RSNeuron = Neuron(
    parameters = """
        a = 0.02 : population
        b = 0.2 : population
        c = -65. : population
        d = 8. : population
        tau_ampa = 5. : population
        tau_nmda = 150. : population
        vrev = 0.0 : population
    """ ,
    equations="""
        # Inputs
        I = g_ampa * (vrev - v) + g_nmda * nmda(v, -80.0, 60.0) * (vrev -v)  
        # Midpoint scheme      
        dv/dt = (0.04 * v + 5.0) * v + 140.0 - u + I : init=-65., midpoint
        du/dt = a * (b*v - u) : init=-13., midpoint
        # Izhikevich scheme
        # new_v = v + 0.5*(0.04 * v^2 + 5.0 * v + 140.0 - u + I) : init=-65.
        # v = new_v + 0.5*(0.04 * new_v^2 + 5.0 * new_v + 140.0 - u + I) : init=-65.
        # u += a * (b*v - u) : init=-13.
        # Conductances
        tau_ampa * dg_ampa/dt = -g_ampa : exponential
        tau_nmda * dg_nmda/dt = -g_nmda : exponential
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
    """
)

# Input population
inp = PoissonPopulation(100, rates=np.linspace(0.2, 20., 100))

# RS neuron without homeostatic mechanism
pop1 = Population(1, RSNeuron)
pop1.compute_firing_rate(5000.)

# RS neuron with homeostatic mechanism
pop2 = Population(1, RSNeuron)
pop2.compute_firing_rate(5000.)

# Nearest Neighbour STDP
nearest_neighbour_stdp = Synapse(
    parameters="""
        tau_plus = 20. : projection
        tau_minus = 60. : projection
        A_plus = 0.0002 : projection
        A_minus = 0.000066 : projection
        w_max = 0.03 : projection
    """,
    equations = """
        # Traces
        tau_plus  * dltp/dt = -ltp : exponential
        tau_minus * dltd/dt = -ltd : exponential
        # Nearest-neighbour
        w += if t_post >= t_pre: ltp else: - ltd : min=0.0, max=w_max
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
        tau_plus = 20. : projection
        tau_minus = 60. : projection
        A_plus = 0.0002 : projection
        A_minus = 0.000066 : projection
        w_min = 0.0 : projection
        w_max = 0.03 : projection

        # Homeostatic regulation
        alpha = 0.1 : projection
        beta = 1.0 : projection
        gamma = 50. : projection
        Rtarget = 35. : projection
        T = 5000. : projection
    """,
    equations = """
        # Traces
        tau_plus  * dltp/dt = -ltp : exponential
        tau_minus * dltd/dt = -ltd : exponential
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

# Projection without homeostatic mechanism
proj1 = Projection(inp, pop1, ['ampa', 'nmda'], synapse=nearest_neighbour_stdp)
proj1.connect_all_to_all(Uniform(0.01, 0.03))

# Projection with homeostatic mechanism
proj2 = Projection(inp, pop2, ['ampa', 'nmda'], synapse=homeo_stdp)
proj2.connect_all_to_all(weights=Uniform(0.01, 0.03))


compile()

# Record
m1 = Monitor(pop1, 'r')
m2 = Monitor(pop2, 'r')
m3 = Monitor(proj1[0], 'w', period=1000.)
m4 = Monitor(proj2[0], 'w', period=1000.)

# Simulate
T = 1000 # 1000s
simulate(T*1000., True)

# Get the data
data1 = m1.get('r')
data2 = m2.get('r')
data3 = m3.get('w')
data4 = m4.get('w')
print('Mean Firing Rate without homeostasis:', np.mean(data1[:, 0]))
print('Mean Firing Rate with homeostasis:', np.mean(data2[:, 0]))

import matplotlib.pyplot as plt
plt.subplot(311)
plt.plot(np.linspace(0, T, len(data1[:, 0])), data1[:, 0], 'r-', label="Without homeostasis")
plt.plot(np.linspace(0, T, len(data2[:, 0])), data2[:, 0], 'b-', label="With homeostasis")
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
plt.subplot(312)
plt.plot(data3[-1, :], 'r-')
plt.plot(data4[-1, :], 'bx')
axes = plt.gca()
axes.set_ylim([0., 0.035])
plt.xlabel('# neuron')
plt.ylabel('Weights after 1000s')
plt.subplot(313)
plt.imshow(data4.T, aspect='auto', cmap='hot')
plt.xlabel('Time (s)')
plt.ylabel('# neuron')
plt.show()

