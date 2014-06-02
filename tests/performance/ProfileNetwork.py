#
#    A network comparable to Dinkelbach et al. 2012
#
from ANNarchy import *

setup(num_threads=1)
#setup( paradigm = "cuda", num_threads=1 )

# Defining the neurons
InputNeuron = RateNeuron(
    equations="""
        r = 0.2 : init = 0.1 
    """
)

OutputNeuron = RateNeuron(
    equations="""
        r = sum(exc)
    """
)

class ProfileNetwork(Network):

    def __init__(self, neuron_count, connection_count):
        
        print 'Config: n =', neuron_count,' c =', connection_count

        input_pop = Population(geometry=(neuron_count), neuron=InputNeuron)
        output_pop = Population(geometry=(neuron_count), neuron=OutputNeuron)
        
        input_output = Projection( input_pop, output_pop, 'exc').connect_fixed_number_pre(connection_count, 1.0)
        
        Network.__init__(self, input_pop, output_pop, input_output)
        
        
    