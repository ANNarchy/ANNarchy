# Proposition for the new syntax
from ANNarchy4 import *

##############################
### Rate neurons
##############################

# Basic leaky neuron
LeakyNeuron = RateNeuron(
    parameters= 
        """tau = 10.0""",
    variables = 
        """ noise = Uniform(-0.1, 0.1) : init = 0.0 # one can specify init values after the 0.0. 
            baseline :  init= 0.0 # if a variable is just used for external inputs but not computed, just write its name
            tau * dmp/dt + mp = sum(exc) - sum(inh) +baseline + noise : init=0.0 # first order eqs should be written this way, I need it for the exponential Euler
            rate = my_function(mp) : init = 0.0, min = 0.0 # you can specify min/max after the : too
        """,
    functions = # why not? not urgent but would be nice
        """ my_function(x) = acos( 2*pi* x)
            my_other_function(x, y) = if x > y then x-y else 0.0
        """
)

# Alternative to functions:
#Function(
#    output = "my_function(x) = acos( 2*pi* x)" # The function is generated globally for all objects. This kind of functions can be parsed efficiently by weave or sympy
#)

# Neuron with temporal adaptation of excitatory inputs
ShuntingPhasicNeuron = RateNeuron(
    parameters= '''
        tau =  10.0
        tau_a = 500.0
    ''',
    variables = 
        """ 
            noise = Uniform(-0.1, 0.1) : init = 0.0 
            tau_a * dAexc/dt + Aexc = sum(exc):  init= 0.0
            input = if sum(exc) > 0.1 then pos(sum(exc) - Aexc) else 0.0
            tau * dmp/dt + mp = input - sum(inh) + noise : init=0.0
            rate = mp : init = 0.0, min = 0.0
        """       
)

##############################
### Rate learning rules
##############################

# Covariance learning rule with homeostatic regularization
#===============================================================================
# Covariance = RateSynapse(
#     parameters = 
#         """ eta' : 100.
#             tau_alpha = 10.0
#         """,
#     variables = 
#         """ tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0) : init = 0.0, side=post_only # allows to specify global variables. If the user says nothing, it is local, too bad for him
#         
#             eta * dvalue/dt = ... 
#                 if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) ) ...
#                 then (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - alpha * (post.rate - mean(post.rate))^2 * value ...
#                 else 0.0 : min = 0.0 
#         """,
#     psp = 
#         """ value * pre.rate # only the return value?
#         """
# )
#===============================================================================

##############################
### Spiking neurons
##############################

IF = SpikeNeuron(
    parameters = 
        """ tau = 10.0
            threshold = 30.0
            V_rest = -65.0
            V_exc = -50.0
        """,
    variables = 
        """ tau*dmp/dt + mp = V_rest + g_exc * (mp - V_exc):
            g_exc = 0 : init=0.0 # g_exc is increased by the synapse, must be reset after using it.
        """,
    spike = 
        """ mp > threshold
        """,
    reset = 
        """ mp = V_rest : refractory = 5ms # some models require to keep the mp at rest level for several ms after a spike
        """
)

Izhikevitch = SpikeNeuron(
    parameters = 
        """ a = 0.02
            b = 0.2
            c = -65.0
            d = 2.0
            threshold = 30.0
        """,
    variables = 
        """ dg_exc/dt + g_exc = 0 : init=0.0
            dg_inh/dt + g_inh = 0 : init=0.0
            I = g_exc + g_inh : init=0.0
            dmp/dt = 0.04 * mp * mp + 5*mp + 140 -u + I : init=c
            u = a * (b*mp - u) : init = c*b
        """,
    spike = 
        """ mp > threshold
        """,
    reset = 
        """ mp = c 
            u = u + d 
        """
)


##############################
### Spiking synapses
##############################

#SimpleExcitatory = SpikingSynapse(
#    pre_spike = 
#        """ post.g_exc += value
#        """
#) # Problem: how to generalize the target? post.g_%(target) += value ?

#===============================================================================
# STDP = SpikingSynapse(
#     parameters = 
#         """ tau_pre = 20.0
#             tau_post = tau_pre
#             delta_A_pre = .01
#             delta_A_post = -delta_A_pre * tau_pre / tau_post * 1.05
#         """,
#     pre_spike = 
#         """ A_pre += delta_A_pre
#             value += A_post
#         """,
#     post_spike = 
#         """ A_post += delta_A_post
#             value += A_pre
#         """,
#     variables = 
#         """ tau_pre * dA_pre/dt + A_pre = 0.0   : init = 0.0, event_driven
#             tau_post * dA_post/dt  + A_post = 0.0 : init = 0.0, side=post_only, event_driven
#         """
# )
#===============================================================================

#CompleteSynapse = SimpleExcitatory + STDP # Combines the psp with the learning rule: in practice, just prepends the pre_spike rule of SimpleExc to the one of STDP. Except pre_spike and post_spike, no variable should have the same name. Adding two classes is easy by implementing __add__ in SpikingSynapse
