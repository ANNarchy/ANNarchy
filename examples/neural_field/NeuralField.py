#
#
#    ANNarchy-4 NeuralField
#
#
from ANNarchy4 import *

setup(dt=1.0)
#
# Define the neuron classes
#
Input = RateNeuron(   
parameters="""
    tau = 1.0
    baseline = 0.0
""",
equations="""
    noise = Uniform(-0.5, 0.5)
    tau * dmp / dt + mp = baseline + noise
    rate = pos(mp)
""" 
)

Focus = RateNeuron(
parameters=""" 
    tau = 20.0
    noise = 0.0
    baseline = 0.0
    threshold_min = 0.0
    threshold_max = 1.0
""",
equations="""
    tau * dmp / dt + mp = sum(exc) + sum(inh) + baseline + noise
    rate = if mp > threshold_max : threshold_max else: pos(mp) : init = 0.0
"""
)

nb_neurons = 20

InputPop = Population((nb_neurons, nb_neurons), Input)
FocusPop = Population((nb_neurons, nb_neurons), Focus)

Proj1 = Projection( 
    pre = InputPop, 
    post = FocusPop, 
    target = 'exc'
).connect_one_to_one( weights=1.0 )
                    
Proj2 = Projection(
    pre = FocusPop, 
    post = FocusPop, 
    target = 'inh'     
).connect_dog_optimized(    
    amp_pos=0.2, 
    sigma_pos=0.1, 
    amp_neg=0.1, 
    sigma_neg=0.7                    
)

# Main program
if __name__ == "__main__":

    # Analyse and compile everything, initialize the parameters/variables...
    compile()    

    import pyximport; pyximport.install()
    import BubbleWorld
    
    BubbleWorld.run(InputPop, FocusPop, Proj2)
    
    raw_input()
