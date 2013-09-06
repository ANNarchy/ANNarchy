#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *

#
# Define the neuron classes
#
Input = Neuron(   
    tau = 1.0,
    rate = Variable(init = 0.0)
)

Layer1 = Neuron(   
    tau = 10.0,
    mp = Variable(init=0.0, eq="tau * dmp / dt + mp = sum(exc)"),
    rate = Variable(init=0.0, eq="rate = pos(mp)"),
    order = ['mp', 'rate']
)

Layer2 = Neuron(
    tau = 10.0,
    mp = Variable(init=0.0, eq="tau * dmp / dt + mp = sum(exc) - sum(inh)"),
    rate = Variable(init=0.0, eq="rate = pos(mp)"),
    order = ['mp', 'rate']
)

Oja = Synapse(
    tau = 2000,
    alpha = 8.0,
    value = Variable(init=0.0, eq = "tau * dvalue / dt = pre.rate*post.rate - alpha * post.rate^2 *value"),
)

AntiHebb = Synapse(
    tau = 2000,
    alpha = 0.3,
    value = Variable(init=0.0, eq = "tau * dvalue / dt = pre.rate*post.rate - alpha * post.rate * value", min=0.0)
)

InputPop = Population(geometry=(8,8), neuron=Input)
Layer1Pop = Population(name="Layer1", geometry=(8,8), neuron=Layer1)
Layer2Pop = Population(name="Layer2", geometry=(6,5), neuron=Layer2)

Proj_In_L1 = Projection(pre=InputPop, post=Layer1Pop, target='exc', connector=Connector('One2One', weights=RandomDistribution('constant', [1.0])))
Proj_L1_L2 = Projection(pre=Layer1Pop, post=Layer2Pop, target='exc', synapse=Oja, connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,0.1])))
Proj_L2_L2 = Projection(pre=Layer2Pop, post=Layer2Pop, target='inh', synapse=AntiHebb, connector=Connector('All2All', weights=RandomDistribution('uniform', [0.0,0.1])))

#
# Analyse and compile everything, initialize the parameters/variables...
#
print 'Compiling'
compile()

if __name__ == '__main__':

    vis = Visualization( [ { 'pop':InputPop, 'var': 'rate' }, 
                           { 'pop': Layer2Pop, 'var': 'rate' }, 
                           { 'proj': Proj_L1_L2, 'var': 'value', 'max': 0.3, 'title': 'Layer1->Layer2 : exc' } ] )

    print 'Running the simulation'

    for trial in range(50000):
        bars = np.zeros((8,8))
        
        # appears a vertical bar?
        for i in xrange(8):
            if np.random.rand(1) < 1.0/8.0:
                bars[:,i] = 1.0

        # appears a horizontal bar?
        for i in xrange(8):
            if np.random.rand(1) < 1.0/8.0:
                bars[i,:] = 1.0

        InputPop.rate = bars.reshape(8*8)

        simulate(50)
        vis.render()
