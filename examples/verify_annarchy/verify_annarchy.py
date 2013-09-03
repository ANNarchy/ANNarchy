from ANNarchy4 import *
from math import fabs

#
# global parameters
#
setup(dt=0.1)

#
# Define some simple classes
#
Input = Neuron(   tau = 1.0,
                  test_var=Variable(init=0.1, eq="test_var=0.2"),
                  rate = Variable(init=1.0),
               )

InputPop = Population(name="Input", geometry=(2,2), neuron=Input)

InputProj = Projection(pre=InputPop, 
                       post=InputPop, 
                       target='exc',
                       connector=Connector('One2One', weights=RandomDistribution('constant', [0.0]))
                       )
#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

#
# tests
#
def test_changed_dt_in_pop():
    assert fabs(InputPop.dt - 0.1) < 0.000001

def test_changed_dt_in_dendrite():
    assert fabs(InputProj.dendrites[0].dt - 0.1) < 0.000001
    
def test_direct_access_parameter_tau():
    assert InputPop.tau == 1.0
    
def test_direct_access_variable():
    assert (InputPop.rate == [1.0, 1.0, 1.0, 1.0]).all()
