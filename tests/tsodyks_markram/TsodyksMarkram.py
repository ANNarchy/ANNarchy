from ANNarchy import *


dt=0.25
setup(dt=dt)

LIF = Neuron(
    parameters = """
    tau = 30.0 : population
    I = 15.0
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + I : init=13.5
    """,
    spike = "v > 15.0",
    reset = "v = 13.5",
    refractory = 3.0
)

TsodyksMarkram = Synapse(
    parameters = """
    w=0.0
    tau_rec = 800.0
    tau_I = 3.0
    tau_facil = 0.0
    U = 0.5 
    """,
    equations = """
    dx/dt = z/tau_rec : min=0.0, max=1.0
    dy/dt = -y/tau_I : min=0.0, max=1.0
    dz/dt = y/tau_I - z/tau_rec : init=1.0, min=0.0, max=1.0
    u = if tau_facil > 0.0 : u - dt * u / tau_facil else: U : min=0.0, max=1.0
    """,
    psp = "w * y",
    pre_spike="""
    u  = if tau_facil > 0.0 : u + U * (1.0 - u) else: U
    x -= u*x
    y += u*x
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

# Create projections
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

tau_I = 3.0

proj_ee = Projection(pre=Exc, post=Exc, target='exc', synapse=TsodyksMarkram).connect_fixed_probability(probability=0.1, weights=Normal(Aee, (Aee/2.0), min=0.2*Aee, max=2.0*Aee)) 
proj_ee.U = Normal(Uee, (Uee/2.0), min=0.1, max=0.9)
proj_ee.tau_rec = Normal(tau_rec_ee, (tau_rec_ee/2.0), min=5.0)
proj_ee.tau_facil = -10.0
proj_ee.tau_I = tau_I

proj_ei = Projection(pre=Inh, post=Exc, target='inh', synapse=TsodyksMarkram).connect_fixed_probability(probability=0.1, weights=Normal(Aei, (Aei/2.0), min=0.2*Aei, max=2.0*Aei))
proj_ei.U = Normal(Uei, (Uei/2.0), min=0.1, max=0.9)
proj_ei.tau_rec = Normal(tau_rec_ei, (tau_rec_ei/2.0), min=5.0)
proj_ei.tau_facil = -10.0
proj_ei.tau_I = Normal(tau_I, (tau_I/2.0))

proj_ie = Projection(pre=Exc, post=Inh, target='exc', synapse=TsodyksMarkram).connect_fixed_probability(probability=0.1, weights=Normal(Aie, (Aie/2.0), min=0.2*Aie, max=2.0*Aie))
proj_ie.U = Normal(Uie, (Uie/2.0), min=0.001, max=0.07)
proj_ie.tau_rec = Normal(tau_rec_ie, (tau_rec_ie/2.0), min=5.0)
proj_ie.tau_facil = Normal(tau_facil_ie, (tau_facil_ie/2.0), min=5.0)
proj_ie.tau_I = tau_I

proj_ii = Projection(pre=Inh, post=Inh, target='inh', synapse=TsodyksMarkram).connect_fixed_probability(probability=0.1, weights=Normal(Aii, (Aii/2.0), min=0.2*Aii, max=2.0*Aii))
proj_ii.U = Normal(Uii, (Uii/2.0), min=0.001, max=0.07)
proj_ii.tau_rec = Normal(tau_rec_ii, (tau_rec_ii/2.0), min=5.0)
proj_ii.tau_facil = Normal(tau_facil_ii, (tau_facil_ii/2.0), min=5.0)
proj_ii.tau_I = tau_I


compile()


# Record
Exc.start_record('spike')
Inh.start_record('spike')

# Simulate
duration = 10000.0
simulate(duration, measure_time=True)

# Retrieve recording
data_exc = Exc.get_record()['spike']
data_inh = Inh.get_record()['spike']
spikes_exc = raster_plot(data_exc)
spikes_inh = raster_plot(data_inh)
spikes = np.concatenate((spikes_exc, spikes_inh + [0, 400]), axis=0)

h = histogram(data_exc)

# Plot
from pylab import *
subplot(1,2,1)
plot(dt*spikes[:, 0], spikes[:, 1], '.', markersize=2.0)
xlim((0, duration)); ylim((0,500))
subplot(1,2,2)
plot(dt*np.arange(h.shape[0]), h/400.0, '-')
show()
