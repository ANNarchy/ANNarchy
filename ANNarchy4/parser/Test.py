from Analyser import *
from ANNarchy4.core.Variable import Variable
from ANNarchy4.core.Neuron import RateNeuron, SpikeNeuron
from ANNarchy4.core.Synapse import RateSynapse, SpikeSynapse
from ANNarchy4.core.Projection import Projection
from ANNarchy4.core.Connector import All2All
from ANNarchy4.core.Population import Population
from ANNarchy4.core.Random import Uniform

from pprint import pprint

DefaultNeuron = RateNeuron(
    parameters = """
        tau = 10.0 : population
        baseline = 1.0
    """,
    equations= """
        tau*drate/dt + rate = sum(exc) - sum(inh) + baseline : min=0.0
    """
)
   
Oja = RateSynapse(
    parameters = """
        eta = 10.0 
        tau_alpha = 10.0 : postsynaptic
    """,
    equations = """
        tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0) : postsynaptic
        eta * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min=0.0
    """,
    psp = """
        value * pre.rate
    """
)   

Izhikevitch = SpikeNeuron(
    parameters = 
        """ a = 0.02 : population
            b = 0.2
            c = -65.0
            d = 2.0
            threshold = 30.0
        """,
    equations = 
        """ I = g_exc + g_inh : init=0.0
            dmp/dt = 0.04 * mp^2 + 5*mp + 140 -u + I : init = -65.0
            u = a * (b*mp - u) : init = -13.0
            dg_exc/dt + g_exc = 0 : init=0.0
            dg_inh/dt + g_inh = 0 : init=0.0
        """,
    spike = 
        """ mp > threshold
        """,
    reset = 
        """ mp = c 
            u = u + d 
        """
)
    
def test_neuron(NeuronType, name):
    """Tests the result of parsing on a single neuron definition.

    * NeuronType is the Neuron instance to test.

    * name is a unique population name for the test.
    """
    pop = Population(name=name, neuron=NeuronType, geometry=1)
    variables = pop.generator.neuron_variables
    parser = NeuronAnalyser(
            variables, 
            ['exc', 'inh'],
            name
    )
    parsed_neuron, global_operations = parser.parse()
    if parsed_neuron:
        for name, desc in parsed_neuron.items():
            print 'Initial description: '
            pprint(variables[name])
            if desc['type'] == 'global':
                print name, 'is a population-wise parameter'
                print '\ttype:', desc['cpp_type']
                print '\tdeclaration:', desc['def']
                print '\tinit:', desc['init']
            else:
                print name, 'is a neuron-specific variable'
                print '\ttype:', desc['cpp_type']
                print '\tdeclaration:', desc['def']
                print '\tinit:', desc['init']
                print '\tequation:', variables[name]['var'].eq
                print '\tupdate rule:', desc['cpp']
            print '-'*60
    else:
        _error('Analyser failed')
        exit(0)


def test_synapse(SynapseType, name):
    """Tests the result of parsing on a single synapse definition.

    * SynapseType is the Neuron instance to test.

    * name is a unique population name for the test.
    """
    # Create dummy network
    pop1 = Population(name=name+"1", neuron=DefaultNeuron, geometry=1)
    pop2 = Population(name=name+"2", neuron=DefaultNeuron, geometry=1)
    proj = Projection(
        pre=pop1, post=pop2, target='exc',
        synapse = SynapseType, 
        connector = All2All(weights= Uniform (-0.5, 0.5 ) ) 
    )
    # Analyse
    variables = proj.generator.synapse_variables
    pprint(variables)
    analyser = SynapseAnalyser(variables)
    analyser.targets_post = ['inh', 'exc']
    parsed_synapse, gop = analyser.parse()
    if parsed_synapse:
        for name, desc in parsed_synapse.items():
            print desc.keys()
            if desc['type'] == 'global':
                print name, 'is a projection-wise parameter'
                print '\ttype:', desc['cpp_type']
                print '\tinit:', desc['init']
            else:
                print name, 'is a synapse_specific variable'
                print '\ttype:', desc['cpp_type']
                print '\tinit:', desc['init']
                print '\tequation:', desc['eq']
                print '\tupdate rule:', desc['cpp']
            print '-'*60
    else:
        _error('Analyser failed')
        exit(0)   


if __name__ == '__main__':
    print 'Testing the parser...'

#    print 'Analysing Leaky neuron'
#    print '-'*60
#    test_neuron(DefaultNeuron, 'testleaky')
    
    print '-'*60
    print 'Analysing Oja learning rule'
    print '-'*60
    test_synapse(Oja, 'testsyn')

#    
#    print '-'*60
#    print 'Analysing Izhikevitch neuron'
#    print '-'*60
#    test_neuron(Izhikevitch, 'testizhi')

