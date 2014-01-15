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
data = {
'EPSILON' : 0.000001,
'SYN_TAU' : 4.0,
'NEUR_TAU' : 10.0
}

from ANNarchy4 import *
from math import fabs
import numpy as np

#
# global parameters
#

#
# Define some simple classes
#
Input = RateNeuron(   
    parameters = """
        tau = 'NEUR_TAU' : population
        test_var = 0.1
    """,
    extra_values = data,
    equations = """
        rate = 1.0
    """              
)

TestSynapse = RateSynapse(
    parameters = """
        tau = 'SYN_TAU' : population
        boolVar = True : type = bool, population
        boolVar2 = False : type = bool
        intVar = 1.0 : type = int
    """,
    extra_values = data,
    equations = """
        value = if boolVar then 1.0 else 0.0 : type = float
    """              
)
print TestSynapse

InputPop = Population(name="Input", geometry=(2,2), neuron=Input)
OutputPop = Population(name="Input", geometry=(2,2), neuron=Input)

InputProj = Projection(pre=InputPop, 
                       post=OutputPop, 
                       target='exc',
                       synapse = TestSynapse,
                       connector=All2All(weights=0.2)                       
                       )
                       
#
# Analyse and compile everything, initialize the parameters/variables...
#
compile()

#
#   Test some code generation stuff
#
def test_proj_variable_bool_detection():
    assert ( type(InputProj.dendrites[0].boolVar2) == bool)

def test_proj_variable_bool_definition():
    assert ( type(InputProj.dendrites[0].boolVar) == bool)

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
    assert np.allclose(InputPop.rate, np.ones((2,2)))

def test_pop_access_variable_rate_with_get():
    assert np.allclose(InputPop.get('rate'), np.ones((2,2)))

def test_pop_access_variable_rate_with_get_variable():
    assert np.allclose(InputPop.get_variable('rate'), np.ones((2,2)))

def test_pop_direct_modify1_variable_rate():
    InputPop.rate = 0.1
    assert np.allclose(InputPop.rate, np.ones((2,2))*0.1)

def test_pop_direct_modify2_variable_rate():
    InputPop.rate = np.array([0.2, 0.3, 0.4, 0.5]).reshape(2,2)
    assert np.allclose(InputPop.rate, np.array([0.2, 0.3, 0.4, 0.5]).reshape(2,2))

def test_pop_direct_modify3_variable_rate():
    InputPop.rate = np.ones(InputPop.geometry)
    assert np.allclose(InputPop.rate, np.ones(InputPop.geometry))

def test_pop_direct_access_variable_test_var():
    tmp = np.ones((2,2))*0.1
    assert np.allclose(InputPop.test_var, tmp)
        
def test_pop_access_variable_test_var_with_get():
    tmp = np.ones((2,2))*0.1
    assert np.allclose(InputPop.get('test_var'), tmp)

def test_pop_access_variable_test_var_with_get_variable():
    tmp = (np.ones((2,2)))*0.1
    assert np.allclose(InputPop.get_variable('test_var'), tmp)

def test_pop_direct_modify1_variable_test_var():
    InputPop.test_var = 0.1
    result = (np.ones((2,2)))*0.1
    assert np.allclose(InputPop.test_var, result)

def test_pop_direct_modify2_variable_test_var():
    InputPop.test_var = np.array([0.2, 0.3, 0.4, 0.5]).reshape(2,2)
    assert np.allclose(InputPop.test_var, np.array([0.2, 0.3, 0.4, 0.5]).reshape(2,2))

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

def test_proj_direct_access_variable_value():
    assert np.allclose(InputProj.dendrites[0].value, [0.2])

def test_proj_access_variable_value_with_get():
    tmp = [0.2, 0.2, 0.2, 0.2]
    assert np.allclose(InputProj.dendrites[0].get('value'), tmp)

def test_proj_access_variable_value_with_get_variable():
    tmp = [0.2, 0.2, 0.2, 0.2]
    assert np.allclose(InputProj.dendrites[0].get_variable('value'), tmp)

def test_proj_direct_access_variable_int_variable():
    assert np.allclose(InputProj.dendrites[0].intVar, 1)

def test_proj_direct_modify_variable_int_variable():
    InputProj.dendrites[0].intVar = 2
    assert np.allclose(InputProj.dendrites[0].intVar, 2)

def test_proj_direct_access_bool_variable():
    assert np.allclose(InputProj.dendrites[0].boolVar, [True])

def test_proj_direct_modify_bool_variable():
    InputProj.dendrites[0].boolVar = False
    assert np.allclose(InputProj.dendrites[0].boolVar, [False])
    
#
#   Test time
#
def test_global_get_current_step_single_inc():
    base_time = current_step()
    simulate(1)
    assert (fabs(current_step() - base_time) < 1+EPSILON)
    
def test_global_get_current_step_several_inc():
    base_time = current_step()
    simulate(20)
    assert (fabs(current_step() - base_time) < 20+EPSILON)

def test_get_current_time():
    assert (fabs(current_time() - current_step()*config['dt']) < EPSILON)
    
#
#    Test some equation stuff
#
def test_synapse_eq_bool_var_set_false():
    InputProj.dendrites[0].boolVar = False
    simulate(1)
    assert np.allclose(InputProj.dendrites[0].value, np.zeros((4,1)))
    
def test_synapse_eq_bool_var_set_true():
    InputProj.dendrites[0].boolVar = True
    simulate(1)
    assert np.allclose(InputProj.dendrites[0].value, np.ones((4,1)))
