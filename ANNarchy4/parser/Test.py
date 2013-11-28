from Analyser import *
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Neuron import Neuron
from ANNarchy4.core.Synapse import Synapse
from ANNarchy4.core.Projection import Projection
from ANNarchy4.core.Connector import Connector
from ANNarchy4.core.Population import Population
from ANNarchy4.core.Random import Uniform

DefaultNeuron = Neuron(
    baseline =  Variable(init=0.1),
    tau = 10.0,
    rate = Variable(init=0.0, eq='tau*drate/dt + rate = sum(exc) + baseline', min=0.0),
)
    
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


def test_synapse(SynapseType, name):
    """Tests the result of parsing on a single synapse definition.

    * SynapseType is the Neuron instance to test.

    * name is a unique population name for the test.
    """
    pop1 = Population(name=name+"1", neuron=DefaultNeuron, geometry=1)
    pop2 = Population(name=name+"2", neuron=DefaultNeuron, geometry=1)
    proj = Projection(
        pre=pop1, post=pop2, target='exc',
        synapse = Oja, 
        connector=Connector('All2All', weights= Uniform (-0.5, 0.5 ) ) 
    )
    variables = proj.generator.synapse_variables
    analyser = SynapseAnalyser(variables)
    analysed, gop = analyser.parse()
    if not analysed:
        print 'Parsing not successful'
    else:
        print '-'*80
        for var in analysed:
            print var['name'], 'is a', var['type'], ', initialized with', var['init']
            if (var['type'] == 'local' or var['type'] == 'global' ) and 'eq' in var.keys():
                print 'Its update rule \"', var['eq'], '\" is translated to C++ with:'
                print var['cpp']
            print '-'*80

if __name__ == '__main__':
    print 'Testing the parser...'

    print 'Leaky neuron'
    LeakyNeuron = Neuron(
        baseline =  Variable(init=0.1),
        tau = 10.0,
        rate = Variable(init=0.0, eq='tau*drate/dt + rate = sum(exc) + baseline', min=0.0),
    )
    test_neuron(LeakyNeuron, 'testleaky')
    
    print 'Oja learning rule'
    Oja = Synapse(
        eta = 10.0,
        alpha = 1.0,
        value = Variable(init=0.0, eq="dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value", min=0.0)
    )
    test_synapse(Oja, 'testsyn')

