from ANNarchy import *

dt = 0.1
setup(dt=dt)

EIF = Neuron(
    parameters = """
    v_rest = -70.0 : population
    cm = 0.2 : population
    tau_m = 10.0 : population
    tau_syn_E = 5.0 : population
    tau_syn_I = 10.0 : population
    e_rev_E = 0.0 : population
    e_rev_I = -80.0 : population
    delta_T = 3.0 : population
    v_thresh = -55.0 : population
    v_reset = -70.0 : population
    v_spike = -20.0 : population
""",
    equations="""
    
    cm * dv/dt = cm/tau_m*(v_rest - v +  delta_T * exp( (v-v_thresh)/delta_T) ) + g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v)  : init=-70.0
    
    tau_syn_E * dg_exc/dt = - g_exc 
    tau_syn_I * dg_inh/dt = - g_inh 
""",
    spike = """
    v > v_spike
""",
    reset = """
    v = v_reset
""",
    refractory = 2.0

)

# Global population
P = Population(geometry=4000, neuron=EIF)

# Subpopulations
Pe = P[:3200]
Pi = P[3200:]
Pe_input = P[:200]
Pi_input = P[3200:3400]

# Projections
we = 0.0015 # excitatory synaptic weight
wi = 0.00375 # inhibitory synaptic weight
Ce = Projection(Pe, P, 'exc').connect_fixed_probability(weights=we, probability=0.05)
Ci = Projection(Pi, P, 'inh').connect_fixed_probability(weights=wi, probability=0.05)

# Initialization
P.v = -70.0 + 10.0 * np.random.rand(P.size)
P.g_exc = (np.random.randn(P.size) * 2.0 + 5.0) * we
P.g_inh = (np.random.randn(P.size) * 2.0 + 5.0) * wi

# Poisson inputs
i_exc = PoissonPopulation(geometry=200, rates="if t < 200.0 : 2000.0 else : 0.0")
i_inh = PoissonPopulation(geometry=200, rates="if t < 100.0 : 2000.0 else : 0.0")
Ie = Projection(i_exc, Pe_input, 'exc').connect_one_to_one(weights=we)
Ii = Projection(i_inh, Pi_input, 'exc').connect_one_to_one(weights=we)

# Compile the Network
compile()

# Simulate
P.start_record(['spike'])

print 'Start simulation'
simulate(250.0, measure_time=True)
#simulate(50.0, measure_time=True)

# Retrieve recordings
data = P.get_record()
spikes = raster_plot(data['spike'])
if len(spikes) == 0 : # Nothing to plot
    exit()

# Plot
from pylab import *
plot(dt*spikes[:, 0], spikes[:, 1], '.')
show()
