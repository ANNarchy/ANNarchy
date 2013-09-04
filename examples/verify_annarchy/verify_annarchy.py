#
#    Test suite for ANNarchy-4.0
#    --------------------------------
#
#    This file should test the simple access functions provided by ANNarchy4.
#
#    Attention:
#
#        The tests reveal no errors in computational evaluation of the network.
#
#    Run:
#
#        python -m pytest -v verify_annarchy.py 
#
#    author: hdin
#
EPSILON = 0.000001
DT = 0.1
SYN_TAU = 4.0
NEUR_TAU = 10.0

from ANNarchy4 import *
from math import fabs
import numpy as np

#
# global parameters
#
setup(dt = DT)

#
# Define some simple classes
#
Input = Neuron(   tau = NEUR_TAU,
                  test_var=Variable(init=0.1),
                  rate = Variable(init=1.0),
               )

TestSynapse = Synapse( tau = SYN_TAU )
 
InputPop = Population(name="Input", geometry=(2,2), neuron=Input)

InputProj = Projection(pre=InputPop, 
                       post=InputPop, 
                       target='exc',
                       synapse = TestSynapse,
                       connector=Connector('One2One', weights=RandomDistribution('constant', [0.2]))                       
                       )
#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

#
#   Test global parameters:
#
def test_pop_changed_dt_in_pop():
    assert fabs(InputPop.dt - DT) < EPSILON

def test_proj_changed_dt_in_dendrite():
    assert fabs(InputProj.dendrites[0].dt - DT) < EPSILON
    
#
#   Test population functions
#
def test_pop_direct_access_parameter_tau():
    assert (fabs(InputPop.tau - NEUR_TAU) < EPSILON)

def test_pop_access_parameter_tau_with_get():
    assert (fabs(InputPop.get('tau') - NEUR_TAU) < EPSILON)

def test_pop_access_parameter_tau_with_get_parameter():
    assert (fabs(InputPop.get_parameter('tau') - NEUR_TAU) < EPSILON)
    
def test_pop_direct_access_variable_rate():
    assert np.allclose(InputPop.rate, [1.0, 1.0, 1.0, 1.0])

def test_pop_access_variable_rate_with_get():
    assert np.allclose(InputPop.get('rate'), np.ones((2,2)))

def test_pop_access_variable_rate_with_get_variable():
    assert np.allclose(InputPop.get_variable('rate'), np.ones((2,2)))

def test_pop_direct_modify1_variable_rate():
    InputPop.rate = 0.1
    assert np.allclose(InputPop.rate, [0.1, 0.1, 0.1, 0.1])

def test_pop_direct_modify2_variable_rate():
    InputPop.rate = [0.2, 0.3, 0.4, 0.5]
    assert np.allclose(InputPop.rate, [0.2, 0.3, 0.4, 0.5])

def test_pop_direct_modify3_variable_rate():
    InputPop.rate = np.ones(InputPop.geometry)
    assert np.allclose(InputPop.rate, np.ones(InputPop.geometry))

def test_pop_direct_access_variable_test_var():
    tmp = [0.1, 0.1, 0.1, 0.1]
    assert np.allclose(InputPop.test_var, tmp)
        
def test_pop_access_variable_test_var_with_get():
    tmp = np.ones((2,2))*0.1
    assert np.allclose(InputPop.get('test_var'), tmp)

def test_pop_access_variable_test_var_with_get_variable():
    tmp = (np.ones((2,2)))*0.1
    assert np.allclose(InputPop.get_variable('test_var'), tmp)

def test_pop_direct_modify1_variable_test_var():
    InputPop.test_var = 0.1
    assert np.allclose(InputPop.test_var, [0.1, 0.1, 0.1, 0.1])

def test_pop_direct_modify2_variable_test_var():
    InputPop.test_var = [0.2, 0.3, 0.4, 0.5]
    assert np.allclose(InputPop.test_var, [0.2, 0.3, 0.4, 0.5])

def test_pop_direct_modify3_variable_test_var():
    InputPop.test_var = np.ones(InputPop.geometry)
    assert np.allclose(InputPop.test_var, np.ones(InputPop.geometry))

#
#   Test projection functions
#
def test_proj_direct_access_parameter_tau():
    assert InputProj.dendrites[0].tau == SYN_TAU

def test_proj_access_parameter_tau_with_get():
    assert InputProj.dendrites[0].get('tau') == SYN_TAU

def test_proj_access_parameter_tau_with_get_parameter():
    assert InputProj.dendrites[0].get_parameter('tau') == SYN_TAU

def test_proj_direct_access_parameter_dt():
    assert fabs(InputProj.dendrites[0].dt - DT) < EPSILON

def test_proj_access_parameter_dt_with_get():
    assert fabs(InputProj.dendrites[0].get('dt') - DT) < EPSILON

def test_proj_access_parameter_dt_with_get_parameter():
    assert fabs(InputProj.dendrites[0].get_parameter('dt') - DT) < EPSILON
    
def test_proj_direct_access_variable_value():
    assert np.allclose(InputProj.dendrites[0].value, [0.2])

def test_proj_access_variable_value_with_get():
    tmp = np.zeros((2,2))
    tmp[0,0] = 0.2
    assert np.allclose(InputProj.dendrites[0].get('value'), tmp)

def test_proj_access_variable_value_with_get_variable():
    tmp = np.zeros((2,2))
    tmp[0,0] = 0.2
    assert np.allclose(InputProj.dendrites[0].get_variable('value'), tmp)
