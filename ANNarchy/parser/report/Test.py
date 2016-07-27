from ANNarchy import *

DefaultNeuron = Neuron(
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
        r = pos(mp) 
    """
)
   
Oja = Synapse(
    parameters = """
        eta = 10.0 
        tau = 10.0 : postsynaptic
    """,
    equations = """
        tau * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
        eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
    """,
    psp = """
        w * pre.r
    """
) 

pop1 = Population(name='pop1', neuron=DefaultNeuron, geometry=1)
pop2 = Population(name='pop2', neuron=DefaultNeuron, geometry=1)
proj = Projection(
    pre=pop1, post=pop2, target='exc',
    synapse = Oja, 
).connect_all_to_all(weights= Uniform (-0.5, 0.5 ) ) 

# Call only the analyser
report()
