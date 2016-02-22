"""
Homeostatic STDP mechanism.

Reimplementation of the model published in:

Carlson, K.D.; Richert, M.; Dutt, N.; Krichmar, J.L., "Biologically plausible models of homeostasis and STDP: Stability and learning in spiking neural networks," in Neural Networks (IJCNN), The 2013 International Joint Conference on , vol., no., pp.1-8, 4-9 Aug. 2013. doi: 10.1109/IJCNN.2013.6706961

Based on the Carlsim tutorial:

http://www.socsci.uci.edu/~jkrichma/CARLsim/doc/tut3_plasticity.html

"""
from ANNarchy import *

# Izhikevich RS neuron
RSNeuron = Neuron(
    parameters = """
        a = 0.02
        b = 0.2
        c = -65.
        d = 8.
        tau_ampa = 5.
        tau_nmda = 150.
        vrev = 0.0
    """ ,
    equations="""
        I = g_ampa * (vrev - v) + g_nmda * nmda(v, -80.0, 60.0) * (vrev -v)        
        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I : init=-65., midpoint
        du/dt = a * (b*v - u) : init=-13.
        tau_ampa * dg_ampa/dt = -g_ampa
        tau_nmda * dg_nmda/dt = -g_nmda
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

# Regular STDP
stdp = Synapse(
    parameters="""
        tau_plus = 20. : postsynaptic
        tau_minus = 60. : postsynaptic
        A_plus = 0.0002 : postsynaptic
        A_minus = 0.000066 : postsynaptic
        w_min = 0.0 : postsynaptic
        w_max = 0.03 : postsynaptic
    """,
    equations = """
        tau_plus  * dltp/dt = -ltp : event-driven
        tau_minus * dltd/dt = -ltd : event-driven
    """,
    pre_spike="""
        g_target += w
        ltp += A_plus 
        w = clip(w - ltd, w_min , w_max)
    """,         
    post_spike="""
        ltd += A_minus 
        w = clip(w + ltp, w_min , w_max)
    """
)

# STDP with homeostatic regulation
homeo_stdp = Synapse(
    parameters="""
        # STDP
        tau_plus = 20. : postsynaptic
        tau_minus = 60. : postsynaptic
        A_plus = 0.0002 : postsynaptic
        A_minus = 0.000066 : postsynaptic
        w_min = 0.0 : postsynaptic
        w_max = 0.03 : postsynaptic

        # Homeostatic regulation
        alpha = 0.1 : postsynaptic
        beta = 1.0 : postsynaptic
        gamma = 50. : postsynaptic
        Rtarget = 35. : postsynaptic
        T = 5000. : postsynaptic
    """,
    equations = """
        tau_plus  * dltp/dt = -ltp
        tau_minus * dltd/dt = -ltd
        R = post.r
        K = R/(T*(1.+fabs(1. - R/Rtarget) * gamma))
        dw/dt = (alpha * w * (1- R/Rtarget) + beta * (ltp-ltd) ) * K : min=w_min, max=w_max
    """,
    pre_spike="""
        g_target += w
        ltp += A_plus 
    """,         
    post_spike="""
        ltd += A_minus 
    """ 
)

# Projection without homeostatic mechanism
proj1 = Projection(inp, pop1, ['ampa', 'nmda'], synapse=stdp)
proj1.connect_all_to_all(Uniform(0.01, 0.03))

# Projection with homeostatic mechanism
proj2 = Projection(inp, pop2, ['ampa', 'nmda'], synapse=homeo_stdp)
proj2.connect_all_to_all(weights=Uniform(0.01, 0.03))


compile()

# Record
m1 = Monitor(pop1, ['r'])
m2 = Monitor(pop2, ['r'])

# Simulate
T = 1000 # 1000s
simulate(T*1000.)

# Get the data
data1 = m1.get('r')
data2 = m2.get('r')
w1 = proj1[0].w
w2 = proj2[0].w
print('Mean Firing Rate without homeostasis:', np.mean(data1[:, 0]))
print('Mean Firing Rate with homeostasis:', np.mean(data2[:, 0]))

from pylab import *
subplot(211)
plot(np.linspace(0, T, len(data1[:, 0])), data1[:, 0], 'r-', label="Without homeostasis")
plot(np.linspace(0, T, len(data2[:, 0])), data2[:, 0], 'b-', label="With homeostasis")
xlabel('Time (s)')
ylabel('Firing rate (Hz)')
subplot(212)
plot(w1, 'r-')
plot(w2, 'bx')
xlabel('# neuron')
ylabel('Weight')
show()

