from ANNarchy import *

dt=0.25
setup(dt=dt)

LIF = Neuron(
    parameters = """
    tau = 30.0 : population
    I = 15.0
    tau_I = 3.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + I : init=13.5
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v > 15.0",
    reset = "v = 13.5",
    refractory = 3.0
)


STP = Synapse(
    parameters = """
    w=0.0
    tau_rec = 1.0
    tau_facil = 1.0
    U = 0.1
    """,
    equations = """
    dx/dt = (1 - x)/tau_rec : init = 1.0
    du/dt = (U - u)/tau_facil : init = 0.1
    """,
    pre_spike="""
    g_target += w * u * x
    x *= (1 - u)
    u += U * (1 - u)
    """
)

# Create populations
Exc = Population(geometry=400, neuron=LIF)
Exc.I = np.sort(Uniform(14.625, 15.375).get_values(400))
Exc.v = Uniform(0.0, 15.0)

Inh = Population(geometry=100, neuron=LIF)
Inh.refractory = 2.0
Inh.I = np.sort(Uniform(14.625, 15.375).get_values(100))
Inh.v = Uniform(0.0, 15.0)

# Parameters for the synapses
Aee = 1.8
Aei = 5.4
Aie = 7.2
Aii = 7.2

Uee = 0.5
Uei = 0.5
Uie = 0.04
Uii = 0.04

tau_rec_ee = 800.0
tau_rec_ei = 800.0
tau_rec_ie = 100.0
tau_rec_ii = 100.0

tau_facil_ie = 1000.0
tau_facil_ii = 1000.0

# Create projections
proj_ee = Projection(pre=Exc, post=Exc, target='exc', synapse=STP).connect_fixed_probability(probability=0.1, weights=Normal(Aee, (Aee/2.0), min=0.2*Aee, max=2.0*Aee)) 
proj_ee.U = Normal(Uee, (Uee/2.0), min=0.1, max=0.9)
proj_ee.tau_rec = Normal(tau_rec_ee, (tau_rec_ee/2.0), min=5.0)
proj_ee.tau_facil = dt

proj_ei = Projection(pre=Inh, post=Exc, target='inh', synapse=STP).connect_fixed_probability(probability=0.1, weights=Normal(Aei, (Aei/2.0), min=0.2*Aei, max=2.0*Aei))
proj_ei.U = Normal(Uei, (Uei/2.0), min=0.1, max=0.9)
proj_ei.tau_rec = Normal(tau_rec_ei, (tau_rec_ei/2.0), min=5.0)
proj_ei.tau_facil = dt

proj_ie = Projection(pre=Exc, post=Inh, target='exc', synapse=STP).connect_fixed_probability(probability=0.1, weights=Normal(Aie, (Aie/2.0), min=0.2*Aie, max=2.0*Aie))
proj_ie.U = Normal(Uie, (Uie/2.0), min=0.001, max=0.07)
proj_ie.tau_rec = Normal(tau_rec_ie, (tau_rec_ie/2.0), min=5.0)
proj_ie.tau_facil = Normal(tau_facil_ie, (tau_facil_ie/2.0), min=5.0)

proj_ii = Projection(pre=Inh, post=Inh, target='inh', synapse=STP).connect_fixed_probability(probability=0.1, weights=Normal(Aii, (Aii/2.0), min=0.2*Aii, max=2.0*Aii))
proj_ii.U = Normal(Uii, (Uii/2.0), min=0.001, max=0.07)
proj_ii.tau_rec = Normal(tau_rec_ii, (tau_rec_ii/2.0), min=5.0)
proj_ii.tau_facil = Normal(tau_facil_ii, (tau_facil_ii/2.0), min=5.0)


compile()


# Record
Exc.start_record('spike')
Inh.start_record('spike')

# Simulate
duration = 10000.0
simulate(duration, measure_time=True)

# Retrieve recordings
data_exc = Exc.get_record()
data_inh = Inh.get_record()
spikes_exc = raster_plot(data_exc['spike'])
spikes_inh = raster_plot(data_inh['spike'])
spikes = np.concatenate((spikes_exc, spikes_inh + [0, 400]), axis=0)

# Histogramm of the exc population
h = histogram(data_exc['spike'], binsize=1.0)

# Mean firing rate in the exc population
rates = []
for neur in data_exc['spike']['data']:
    rates.append(len(neur)/duration*1000.0)

# Plot
from pylab import *
subplot(3,1,1)
plot(dt*spikes[:, 0], spikes[:, 1], '.', markersize=1.0)
xlim((0, duration)); ylim((0,500))
subplot(3,1,2)
plot(h/400.0, '-')
subplot(3,1,3)
plot(sorted(rates), '-')
show()
