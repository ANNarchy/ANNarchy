from Analyser import *
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Neuron import Neuron
from ANNarchy4.core.Population import Population

def test_neuron(NeuronType, name):
    """Tests the result of parsing on a single neuron definition.

    * NeuronType is the Neuron instance to test.

    * name is a unique population name for the test.
    """
    pop = Population(name=name, neuron=NeuronType, geometry=1)
    variables = pop.generator.neuron_variables
    analyser = NeuronAnalyser(variables, pop.generator.targets)
    analysed, gop = analyser.parse()
    if not analysed:
        print 'Parsing not successful'
    else:
        print '-'*80
        for var in analysed:
            print var['name'], 'is a', var['type'], ', defined with', var['def'], 'and initialized with', var['init']
            if var['type'] == 'variable' and 'eq' in var.keys():
                print 'Its update rule \"', var['eq'], '\" is translated to C++ with:'
                print var['cpp']
            print '-'*80

if __name__ == '__main__':
    print 'Testing the parser...'

    print 'Leaky neuron'
    LeakyNeuron = Neuron(
        baseline =  Variable(init=0.0),
        tau = 10.0,
        rate = Variable(init=0.0, eq='tau*drate/dt + rate = sum(exc) + baseline', min=0.0),
    )
    test_neuron(LeakyNeuron, 'test1')

