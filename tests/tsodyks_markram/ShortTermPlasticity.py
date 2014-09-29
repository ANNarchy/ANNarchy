from ANNarchy import *


dt=0.25
setup(dt=dt)


InputNeuron = Neuron(
    parameters = "rate=0.0",
    equations="dx/dt= rate/1000.0",
    spike= "x>1.0",
    reset="x=0.0"
)

LIF = Neuron(
    parameters = """
    tau = 10.0 : population
    tau_e = 3.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc 
    tau_e * dg_exc/dt = -g_exc
    """,
    spike = "v > 15.0",
    reset = "v = 0.0",
)

STP = Synapse(
    parameters = """
    w=0.0
    tau_d = 1.0
    tau_f = 100.0
    U = 0.1
    """,
    equations = """
    dx/dt = (1 - x)/tau_d : init = 1.0
    du/dt = (U - u)/tau_f : init = 0.1
    """,
    pre_spike="""
    g_target += w * u * x
    x *= (1 - u)
    u += U * (1 - u)
    """
)

Input = Population(geometry=10, neuron=InputNeuron)
Input.rate = np.linspace(5.0, 30.0, 10)

Output = Population(geometry=10, neuron=LIF)

proj = Projection(pre=Input, post=Output, target= 'exc', synapse=STP).connect_one_to_one(weights=25.0)

# Facilitating
proj.tau_d = 1.0
proj.tau_f = 100.0
proj.U = 0.1

# Depressing
proj.tau_d = 100.0
proj.tau_f = 10.0
proj.U = 0.6

compile()

Input.start_record('spike')
Output.start_record('v')
simulate(1000.0)

input_spikes = raster_plot(Input.get_record()['spike'])

v = Output.get_record()['v']['data']

# Plot
from pylab import *
subplot(2,1,1)
plot(v[0, :])
subplot(2,1,2)
plot(v[9, :])
show()

