#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *
from datetime import datetime

setup(suppress_warnings=True)

#
# Define the neuron classes
#
Input = Neuron(   
    rate = Variable(init = 0.0),
)

Output = Neuron(   
    tau = 1.0,
    mp1 = Variable(init = 0, eq="mp1 = sum(exc)"),
    mp2 = Variable(init = 0, eq="mp2 = sum(exc2)"),
    rate = Variable(init = 0.0, eq="tau * drate/dt +rate = pos(mp1) + pos(mp2)"),
    order = ['mp1', 'mp2', 'rate']
)

class MyConnector(Connector):

    def __init__(self, weights, delays=0, **parameters):
        Connector.__init__(self, weights, delays, **parameters)

    def connect(self):

        pre = self.proj.pre
        post = self.proj.post

        dendrites=[]
        post_ranks=[]
        
        for n in xrange(post.size):
            dendrites.append(Dendrite(self.proj, post_rank=n, ranks=[n], weights=self.weights.get_values(1), delays=self.delays.get_values(1)))
            post_ranks.append(n)

        return dendrites, post_ranks

InputPop = Population(name="Input", geometry=(8,1), neuron=Input)
Layer1Pop = Population(name="Layer1", geometry=(8,1), neuron=Output)


Proj = Projection(pre="Input", post="Layer1", target='exc', connector=MyConnector(weights=1.0, delays=2.0))
Proj2 = Projection(pre="Input", post="Layer1", target='exc2', connector=One2One(weights=1.0, delays=2.0))

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

def synapse_test():

    InputPop.rate = Uniform(0,1).get_values(InputPop.geometry)

    for i in range(5):
        simulate(1)
	print 'One2One'
        print Layer1Pop.mp1
	print 'MyConnector'
        print Layer1Pop.mp2

    #print InputPop.rate
    #print Layer1Pop.rate

if __name__ == '__main__':

    synapse_test()
