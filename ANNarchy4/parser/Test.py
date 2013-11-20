from Analyser import *
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Neuron import Neuron
from ANNarchy4.core.Population import Population

LeakyNeuron = Neuron(
    baseline =  0.0,
    tau = 10.0,
    rate = Variable(init=0.0, eq='tau*drate/dt - rate - 1 = sum(exc) + baseline'),
)

def print_parsing(NeuronType):
    """Prints the result of parsing on a single neuron definition"""
    pop = Population(name='test', neuron=NeuronType, geometry=1)
    analyser = NeuronAnalyser(pop.generator.neuron_variables, pop.generator.targets)
    analysed, gop = analyser.parse()
    if not analysed:
        print 'Parsing not successful'
    else:
        for var in analysed:
            print var['name'], 'is a', var['type'], ', initialized with', var['init']
            if var['type'] == 'variable':
                print var['cpp']
            print '-'*80

if __name__ == '__main__':
    print 'Testing the parser...'
    print_parsing(LeakyNeuron)
