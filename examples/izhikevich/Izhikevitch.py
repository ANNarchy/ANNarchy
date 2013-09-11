from ANNarchy4 import *

setup()

# Define the neurons
Izhikevitch = Neuron(
    baseline = -65.0,
    a = Variable(init=0.02),
    b = Variable(init=0.2),
    c = Variable(init=-65.0),
    d = Variable(init=2.0),
    noise = 1.0,
    I = Variable(init=0.0, eq="I = noise * RandomDistribution('uniform', [0,1]) + sum(exc) - sum(inh)"),
    mp = Variable(init=-65., eq="dmp/dt = 0.04 * mp * mp + 5*mp + 140 -u +I"),
    u = Variable(init=-65.*0.2, eq="u = a * (b*mp - u)"), # init should be b*baseline
    rate = Variable(init=0.0, eq="rate = if mp >30 then 1 else 0"), # TODO : reset the other variables!!! mp=c, u=u+d
    order = ['I', 'mp', 'u', 'rate']
)

# Create the populations
Excitatory = Population(name="Excitatory", geometry=(40, 20), neuron=Izhikevitch)
Excitatory.a = 0.02
Excitatory.b = 0.2
Excitatory.c = RandomDistribution('uniform', [-65., -50.])
Excitatory.d = RandomDistribution('uniform', [2., 8.])
Excitatory.noise = 5.0

Inhibitory = Population(name="Inhibitory", geometry=(20, 10), neuron=Izhikevitch)
Inhibitory.a = RandomDistribution('uniform', [0.02, 1.])
Inhibitory.b = RandomDistribution('uniform', [0.2, 0.25])
Inhibitory.c = -65.
Inhibitory.d = 2.0
Inhibitory.noise = 2.0

# Connect the populations
exc_exc = Projection(pre=Excitatory, post=Excitatory, target='exc',
                     connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0, 0.5] ))
                    )
exc_inh = Projection(pre=Excitatory, post=Inhibitory, target='exc',
                     connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0, 0.5] ))
                    )
inh_exc = Projection(pre=Inhibitory, post=Excitatory, target='inh',
                     connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0, 1.0] ))
                    )
inh_inh = Projection(pre=Inhibitory, post=Inhibitory, target='inh',
                     connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0, 1.0] ))
                    )

# Compile
compile()

# Run the simulation
simulate(1000)
