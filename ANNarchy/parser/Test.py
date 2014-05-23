from ANNarchy import *

DefaultNeuron = RateNeuron(
    parameters = """
        tau = 10.0 : population
        baseline = 1.0
        condition = True 
    """,
    equations= """
        noise = Uniform(-0.1, 0.1)
        tau * dmp/dt + mp = sum(exc) 
                            - sum(inh) 
                            + baseline + noise : implicit
        rate = pos(mp) 
    """
)
   
Oja = RateSynapse(
    parameters = """
        eta = 10.0 
        tau = 10.0 : postsynaptic
    """,
    equations = """
        tau * dalpha/dt + alpha = pos(post.rate - 1.0) : postsynaptic
        eta * dw/dt = pre.rate * post.rate - alpha * post.rate^2 * w : min=0.0
    """,
    psp = """
        w * pre.rate
    """
) 

pop1 = Population(name='pop1', neuron=DefaultNeuron, geometry=1)
pop2 = Population(name='pop2', neuron=DefaultNeuron, geometry=1)
proj = Projection(
    pre=pop1, post=pop2, target='exc',
    synapse = Oja, 
).connect_all_to_all(weights= Uniform (-0.5, 0.5 ) ) 

# Call only the analyser
from ANNarchy.core.Global import _populations, _projections
from ANNarchy.parser.Analyser import Analyser
from pprint import pprint

analyser = Analyser(_populations, _projections)
analyser.analyse()
pprint(analyser.analysed_populations)
pprint(analyser.analysed_projections)
