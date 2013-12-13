from ANNarchy4 import *

MyNeuron = Neuron( 
    tau = 10.0, 
    baseline = Variable(init=0.0),
    bar = Variable(init=0.0),
    foo = Variable(init=0.0),
    rate = Variable(init=0.0, eq="tau * drate/dt + rate = baseline - (bar - foo)", min=0.0)
)

#Oja = Synapse(
#    tau = 2000,
#    alpha = 8.0,
#    value = Variable(init=0.0, eq = "tau * dvalue / dt = pre.rate*post.rate - alpha * post.rate^2 *value"),
#)

input_pop = Population(name = "Input", geometry=(10, 10), neuron=MyNeuron)
#output_pop = Population(name = "Output", geometry=(10, 10), neuron=MyNeuron)

#proj = Projection(pre=input_pop, post=output_pop, target='exc', 
#                  connector=Connector('One2One', weights=RandomDistribution('constant', [1.0])) )

compile()
