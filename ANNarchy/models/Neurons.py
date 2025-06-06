"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Neuron import Neuron

def list_standard_neurons():
    "Returns a list of standard neuron models available."
    return [
        LeakyIntegrator, 
        Izhikevich, 
        IF_curr_exp, 
        IF_cond_exp, 
        IF_curr_alpha, 
        IF_cond_alpha, 
        HH_cond_exp, 
        EIF_cond_alpha_isfa_ista, 
        EIF_cond_exp_isfa_ista,
    ]


##################
### Leaky Integrator
##################

class LeakyIntegrator(Neuron):
    r"""
    Leaky-integrator rate-coded neuron, optionally noisy.

    This simple rate-coded neuron defines an internal variable $v(t)$ which integrates the inputs $I(t)$ with a global time constant $\tau$ and a local baseline $B$. An additive noise $N(t)$ can be optionally defined: 

    $$\tau \cdot \frac{dv(t)}{dt} + v(t) = I(t) + B + N(t)$$

    The transfer function is the positive (or rectified linear ReLU) function with a threshold $T$:

    $$r(t) = (v(t) - T)^+$$

    By default, the input $I(t)$ to this neuron is `"sum(exc) - sum(inh)"`, but this can be changed by  setting the ``sum`` argument:

    ```python
    neuron = ann.LeakyIntegrator(sum="sum(ampa)")
    ```

    By default, there is no additive noise, but the ``noise`` argument can be passed with a specific distribution:

    ```python
    neuron = ann.LeakyIntegrator(noise="Normal(0.0, 1.0)")
    ```

    Equivalent code:

    ```python
    LeakyIntegrator = Neuron(
        parameters=dict(
            tau = 10.0,
            B = ann.Parameter(0.0),
            T = 0.0,
        ), 
        equations=[
            ann.Variable()'tau * dv/dt + v = sum(exc) - sum(inh) + B', method=exponential),
            'r = pos(v - T)',
        ]
    )
    ```

    :param tau: Time constant (global).
    :param B: Baseline (local).
    :param T: Threshold (global).
    :param sum: Input sums.
    """

    # For reporting
    _instantiated = []

    def __init__(self, 
                 tau:float=10.0, 
                 B:float=0.0, 
                 T:float=0.0, 
                 sum:str='sum(exc) - sum(inh)', 
                 noise:str=None) -> None:
        
        # Create the arguments
        parameters = """
            tau = %(tau)s : population
            B = %(B)s
            T = %(T)s : population
        """ % {'tau': tau, 'B': B, 'T': T}

        # Equations for the variables
        if not noise:
            noise_def = ''
        else:
            noise_def = '+ ' + noise

        equations="""
            tau * dv/dt + v = %(sum)s + B %(noise)s : exponential
            r = pos(v - T)
        """ % { 'sum' : sum, 'noise': noise_def}

        Neuron.__init__(self, 
            parameters=parameters, equations=equations,
            name="Leaky-Integrator", 
            description="Leaky-Integrator with positive transfer function and additive noise.")

        # For reporting
        self._instantiated.append(True)

##################
### Izhikevich
##################
class Izhikevich(Neuron):
    '''
    Izhikevich quadratic spiking neuron.

    > Izhikevich, E.M. (2003). *Simple Model of Spiking Neurons, IEEE Transaction on Neural Networks*, 14:6. <http://dx.doi.org/10.1109/TNN.2003.820440>
    
    The neural equations are:

    $$\\frac{dv}{dt} = 0.04 * v^2 + 5.0 * v + 140.0 - u + I$$

    $$\\frac{du}{dt} = a * (b * v - u)$$

    By default, the conductance is "g_exc - g_inh", but this can be changed by setting the `conductance` argument:

    ```python
    neuron = ann.Izhikevich(conductance='g_ampa * (1 + g_nmda) - g_gaba')
    ```

    The synapses are instantaneous, i.e the corresponding conductance is increased from the synaptic efficiency w at the time step when a spike is received.

    The ODEs are solved using the explicit Euler method.

    Equivalent code:

    ```python
    Izhikevich = ann.Neuron(
        parameters = dict(
            noise = ann.Parameter(0.0),
            a = ann.Parameter(0.02),
            b = ann.Parameter(0.2),
            c = ann.Parameter(-65.0),
            d = ann.Parameter(8.0),
            v_thresh = ann.Parameter(30.0),
            i_offset = ann.Parameter(0.0),
        ), 
        equations = [
            'I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset',
            ann.Variable('dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I', init = -65.0),
            ann.Variable('du/dt = a * (b*v - u)', init= -13.0),
        ],
        spike = "v > v_thresh",
        reset = "v = c; u += d",
        refractory = 0.0
    )
    ```

    The default parameters are for a regular spiking (RS) neuron. They are defined as local parameters, so their value can be set at the neuron-level. Here is the neuron definition using global parameters:


    ```python
    Izhikevich = ann.Neuron(
        parameters = dict(
            noise = 0.0,
            a = 0.02,
            b = 0.2,
            c = -65.0,
            d = 8.0,
            v_thresh = 30.0,
            i_offset = 0.0,
        ), 
        equations = [
            'I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset',
            ann.Variable('dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I', init = -65.0),
            ann.Variable('du/dt = a * (b*v - u)', init= -13.0),
        ],
        spike = "v > v_thresh",
        reset = "v = c; u += d",
        refractory = 0.0
    )
    ```

    :param a: Speed of the recovery variable
    :param b: Scaling of the recovery variable
    :param c: Reset potential.
    :param d: Increment of the recovery variable after a spike.
    :param v_thresh: Spike threshold (mV).
    :param i_offset: external current (nA).
    :param noise: Amplitude of the normal additive noise.
    :param tau_refrac: Duration of refractory period (ms).
    :param conductance: Conductances used as inputs.
    '''

    # For reporting
    _instantiated = []
    
    def __init__(self, 
        a:float=0.02, 
        b:float=0.2, 
        c:float=-65.0, 
        d:float=8.0, 
        v_thresh:float=30.0, 
        i_offset:float=0.0, 
        noise:float=0.0, 
        tau_refrac:float=0.0, 
        conductance:str="g_exc - g_inh") -> None:
        
        # Extract which targets are defined in the conductance
        #import re
        #targets = re.findall(r'g_([\w]+)', conductance)
        
        # Create the arguments
        parameters = """
            noise = %(noise)s
            a = %(a)s
            b = %(b)s
            c = %(c)s
            d = %(d)s
            v_thresh = %(v_thresh)s
            i_offset = %(i_offset)s
            tau_refrac = %(tau_refrac)s
        """ % {'a': a, 'b':b, 'c':c, 'd':d, 'v_thresh':v_thresh, 'i_offset':i_offset, 'noise':noise, 'tau_refrac':tau_refrac}
                
        # Equations for the variables
        equations="""
            I = %(conductance)s + noise * Normal(0.0, 1.0) + i_offset
            dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I : init = %(c)s
            du/dt = a * (b*v - u) : init= %(u)s
        """ % { 'conductance' : conductance, 'c':c , 'u': b*c}

        spike = """
            v > v_thresh
        """
        reset = """
            v = c
            u += d
        """
        Neuron.__init__(self, 
            parameters=parameters, equations=equations, spike=spike, reset=reset, refractory='tau_refrac',
            name="Izhikevich", description="Quadratic integrate-and-fire spiking neuron with adaptation.")

        # For reporting
        self._instantiated.append(True)


##################
### IF neurons
##################
class IF_curr_exp(Neuron):
    '''
    Leaky integrate-and-fire model with fixed threshold and decaying-exponential post-synaptic current. 
    
    (Separate synaptic currents for excitatory and inhibitory synapses).

    The ODEs are solved using the exponential Euler method.
    
    Equivalent code:

    ```python
    IF_curr_exp = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-65.0),
            cm  = ann.Parameter(1.0),
            tau_m  = ann.Parameter(20.0),
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0),
            v_thresh = ann.Parameter(-50.0),
            v_reset = ann.Parameter(-65.0),
            i_offset = ann.Parameter(0.0),
        ), 
        equations = [
            ann.Variable(
                'cm * dv/dt = cm/tau_m*(v_rest -v)   + g_exc - g_inh + i_offset',
                method='exponential', init=-65.0
            ),
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),
        ],
        spike = "v > v_thresh",
        reset = "v = v_reset",
        refractory = 0.0
    )
    ```

    :param v_rest:  Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Decay time of excitatory synaptic current (ms)
    :param tau_syn_I: Decay time of inhibitory synaptic current (ms)
    :param v_thresh: Spike threshold (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param i_offset: Offset current (nA)

    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, 
                 v_rest:float=-65.0, cm:float=1.0, 
                 tau_m:float=20.0, tau_refrac:float=0.0, 
                 tau_syn_E:float=5.0, tau_syn_I:float=5.0, 
                 v_thresh:float=-50.0, v_reset:float=-65.0, 
                 i_offset:float=0.0):
        
        # Create the arguments
        parameters = """
            v_rest = %(v_rest)s
            cm  = %(cm)s
            tau_m  = %(tau_m)s
            tau_refrac = %(tau_refrac)s
            tau_syn_E = %(tau_syn_E)s
            tau_syn_I = %(tau_syn_I)s
            v_thresh = %(v_thresh)s
            v_reset = %(v_reset)s
            i_offset = %(i_offset)s
        """ % {'v_rest':v_rest, 'cm':cm, 'tau_m':tau_m, 'tau_refrac':tau_refrac, 
                'tau_syn_E':tau_syn_E, 'tau_syn_I':tau_syn_I, 
                'v_thresh':v_thresh, 'v_reset':v_reset, 'i_offset':i_offset}
                
        # Equations for the variables
        equations="""    
            cm * dv/dt = cm/tau_m*(v_rest -v)   + g_exc - g_inh + i_offset : exponential, init=%(v_reset)s
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
        """ % {'v_reset':v_reset}

        spike = """
            v > v_thresh
        """
        reset = """
            v = v_reset
        """
        Neuron.__init__(
            self, 
            parameters=parameters, 
            equations=equations, 
            spike=spike, 
            reset=reset, 
            refractory='tau_refrac',
            name="Integrate-and-Fire", 
            description="Leaky integrate-and-fire model with fixed threshold and decaying-exponential post-synaptic current.")

        # For reporting
        self._instantiated.append(True)

class IF_cond_exp(Neuron):
    '''
    Leaky integrate-and-fire model with fixed threshold and decaying-exponential post-synaptic conductance.

    The ODEs are solved using the exponential Euler method.

    Equivalent code:

    ```python
    IF_cond_exp = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-65.0),
            cm  = ann.Parameter(1.0),
            tau_m  = ann.Parameter(20.0),
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0),
            e_rev_E = ann.Parameter(0.0),
            e_rev_I = ann.Parameter(-70.0),
            v_thresh = ann.Parameter(-50.0),
            v_reset = ann.Parameter(-65.0),
            i_offset = ann.Parameter(0.0),
        ), 
        equations = [
            ann.Variable(
                'cm * dv/dt = cm/tau_m * (v_rest -v) + g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset',
                method='exponential', init=-65.0
            ),
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),
        ],
        spike = "v > v_thresh",
        reset = "v = v_reset",
        refractory = 0.0
    )
    ```


    :param v_rest:  Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Decay time of excitatory synaptic current (ms)
    :param tau_syn_I: Decay time of inhibitory synaptic current (ms)
    :param e_rev_E: Reversal potential of excitatory conductance (mV)
    :param e_rev_I: Reversal potential of inhibitory conductance (mV)
    :param v_thresh: Spike threshold (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param i_offset: Offset current (nA)
    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refrac=0.0, 
        tau_syn_E=5.0, tau_syn_I=5.0, e_rev_E = 0.0, e_rev_I = -70.0, 
        v_thresh=-50.0, v_reset=-65.0, i_offset=0.0):
        
        # Create the arguments
        parameters = """
            v_rest = %(v_rest)s
            cm  = %(cm)s
            tau_m  = %(tau_m)s
            tau_refrac = %(tau_refrac)s
            tau_syn_E = %(tau_syn_E)s
            tau_syn_I = %(tau_syn_I)s
            v_thresh = %(v_thresh)s
            v_reset = %(v_reset)s
            i_offset = %(i_offset)s
            e_rev_E = %(e_rev_E)s 
            e_rev_I = %(e_rev_I)s
        """ % {'v_rest':v_rest, 'cm':cm, 'tau_m':tau_m, 'tau_refrac':tau_refrac, 
                'tau_syn_E':tau_syn_E, 'tau_syn_I':tau_syn_I, 'v_thresh':v_thresh, 
                'v_reset':v_reset, 'i_offset':i_offset, 'e_rev_E': e_rev_E, 'e_rev_I': e_rev_I}
                
        # Equations for the variables
        equations="""    
            cm * dv/dt = cm/tau_m*(v_rest -v)   + g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset : exponential, init=%(v_reset)s
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
        """% {'v_reset':v_reset}

        spike = """
            v > v_thresh
        """
        reset = """
            v = v_reset
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset, refractory='tau_refrac',
            name="Integrate-and-Fire", 
            description="Leaky integrate-and-fire model with fixed threshold and decaying-exponential post-synaptic conductances.")

        # For reporting
        self._instantiated.append(True)

# Alpha conductances
class IF_curr_alpha(Neuron):
    '''
    Leaky integrate-and-fire model with fixed threshold and alpha post-synaptic currents. 
    
    Separate synaptic currents for excitatory and inhibitory synapses.

    The alpha currents are calculated through a system of two linears ODEs. After a spike is received at t_spike, it peaks at t_spike + tau_syn_X, with a maximum equal to the synaptic efficiency.

    The ODEs are solved using the exponential Euler method.

    Equivalent code:

    ```python
    IF_curr_alpha = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-65.0),
            cm  = ann.Parameter(1.0),
            tau_m  = ann.Parameter(20.0),
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0),
            v_thresh = ann.Parameter(-50.0),
            v_reset = ann.Parameter(-65.0),
            i_offset = ann.Parameter(0.0),
        ), 
        equations = [
            # Scaling
            'gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)',
            'gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)',

            # Membrane potential
            ann.Variable(
                'cm * dv/dt = cm/tau_m*(v_rest -v)   + alpha_exc - alpha_inh + i_offset',
                method='exponential', init=-65.0),

            # Alpha-shaped conductance
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),

            ann.Variable('tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc', method='exponential'),
            ann.Variable('tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh', method='exponential'),
        ],
        spike = "v > v_thresh",
        reset = "v = v_reset",
        refractory = 0.0
    )
    ```

    :param v_rest: Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Rise time of excitatory synaptic current (ms)
    :param tau_syn_I: Rise time of inhibitory synaptic current (ms)
    :param v_thresh: Spike threshold (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param i_offset: Offset current (nA)
    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refrac=0.0, 
        tau_syn_E=5.0, tau_syn_I=5.0, v_thresh=-50.0, v_reset=-65.0, i_offset=0.0):
        
        # Create the arguments
        parameters = f"""
            v_rest = {v_rest}
            cm  = {cm}
            tau_m  = {tau_m}
            tau_refrac = {tau_refrac}
            tau_syn_E = {tau_syn_E}
            tau_syn_I = {tau_syn_I}
            v_thresh = {v_thresh}
            v_reset = {v_reset}
            i_offset = {i_offset}
        """ 
                
        # Equations for the variables
        equations="""  
            gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)
            gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)  
            cm * dv/dt = cm/tau_m*(v_rest -v)   + alpha_exc - alpha_inh + i_offset : exponential, init=%(v_reset)s
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
            tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc  : exponential
            tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh  : exponential
        """  % {'v_reset':v_reset}

        spike = """
            v > v_thresh
        """
        reset = """
            v = v_reset
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset, refractory='tau_refrac',
            name="Integrate-and-Fire", 
            description="Leaky integrate-and-fire model with fixed threshold and alpha post-synaptic currents.")

        # For reporting
        self._instantiated.append(True)


class IF_cond_alpha(Neuron):
    '''
    Leaky integrate-and-fire model with fixed threshold and alpha post-synaptic conductance.
    
    Separate synaptic currents for excitatory and inhibitory synapses.

    The alpha currents are calculated through a system of two linears ODEs. After a spike is received at t_spike, it peaks at t_spike + tau_syn_X, with a maximum equal to the synaptic efficiency.

    The ODEs are solved using the exponential Euler method.

    Equivalent code:

    ```python
    IF_cond_alpha = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-65.0),
            cm  = ann.Parameter(1.0),
            tau_m  = ann.Parameter(20.0),
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0),
            v_thresh = ann.Parameter(-50.0),
            v_reset = ann.Parameter(-65.0),
            i_offset = ann.Parameter(0.0),
        ), 
        equations = [
            # Scaling
            'gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)',
            'gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)',

            # Membrane potential
            ann.Variable(
                'cm * dv/dt = cm/tau_m*(v_rest -v) + alpha_exc  * (e_rev_E - v) + alpha_inh * (e_rev_I - v) + i_offset',
                method='exponential', init=-65.0),

            # Alpha-shaped conductance
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),

            ann.Variable('tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc', method='exponential'),
            ann.Variable('tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh', method='exponential'),
        ],
        spike = "v > v_thresh",
        reset = "v = v_reset",
        refractory = 0.0
    )
    ```


    :param v_rest: Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Rise time of excitatory synaptic current (ms)
    :param tau_syn_I: Rise time of inhibitory synaptic current (ms)
    :param v_thresh: Spike threshold (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param i_offset: Offset current (nA)

    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, v_rest=-65.0, cm=1.0, tau_m=20.0, tau_refrac=0.0, 
        tau_syn_E=5.0, tau_syn_I=5.0, e_rev_E = 0.0, e_rev_I = -70.0, 
        v_thresh=-50.0, v_reset=-65.0, i_offset=0.0):
        
        # Create the arguments
        parameters = """
            v_rest = %(v_rest)s
            cm  = %(cm)s
            tau_m  = %(tau_m)s
            tau_refrac = %(tau_refrac)s
            tau_syn_E = %(tau_syn_E)s
            tau_syn_I = %(tau_syn_I)s
            v_thresh = %(v_thresh)s
            v_reset = %(v_reset)s
            i_offset = %(i_offset)s
            e_rev_E = %(e_rev_E)s 
            e_rev_I = %(e_rev_I)s
        """ % {'v_rest':v_rest, 'cm':cm, 'tau_m':tau_m, 'tau_refrac':tau_refrac, 
                'tau_syn_E':tau_syn_E, 'tau_syn_I':tau_syn_I, 'v_thresh':v_thresh, 
                'v_reset':v_reset, 'i_offset':i_offset, 'e_rev_E': e_rev_E, 'e_rev_I': e_rev_I}
                
        # Equations for the variables
        equations="""    
            gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)
            gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)
            cm * dv/dt = cm/tau_m*(v_rest -v)   + alpha_exc * (e_rev_E - v) + alpha_inh * (e_rev_I - v) + i_offset : exponential, init=%(v_reset)s
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
            tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc  : exponential
            tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh  : exponential
        """ % {'v_reset': v_reset}

        spike = """
            v > v_thresh
        """
        reset = """
            v = v_reset
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset, refractory='tau_refrac',
            name="Integrate-and-Fire", 
            description="Leaky integrate-and-fire model with fixed threshold and alpha post-synaptic conductances.")

        # For reporting
        self._instantiated.append(True)

##################
### EIF neurons
##################

class EIF_cond_exp_isfa_ista(Neuron):
    '''
    Exponential integrate-and-fire neuron with spike triggered and sub-threshold adaptation currents (isfa, ista reps.), decaying-exponential post-synaptic conductances.
     
    Definition according to:

    > Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    Equivalent code:

    ```python

    EIF_cond_exp_isfa_ista = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-70.6),
            cm = ann.Parameter(0.281), 
            tau_m = ann.Parameter(9.3667), 
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0), 
            e_rev_E = ann.Parameter(0.0),
            e_rev_I = ann.Parameter(-80.0),
            tau_w = ann.Parameter(144.0), 
            a = ann.Parameter(4.0),
            b = ann.Parameter(0.0805),
            i_offset = ann.Parameter(0.0),
            delta_T = ann.Parameter(2.0),
            v_thresh = ann.Parameter(-50.4),
            v_reset = ann.Parameter(-70.6),
            v_spike = ann.Parameter(-40.0),
        ), 
        equations = [
            # Input current
            'I = g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset',     
            # Membrane potential     
            ann.Variable('tau_m * dv/dt = (v_rest - v +  delta_T * exp((v-v_thresh)/delta_T)) + tau_m/cm*(I - w)', init=-70.6),     
            # Recovery variable 
            'tau_w * dw/dt = a * (v - v_rest) / 1000.0 - w',       
            # Conductances
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),
        ],
        spike = "v > v_spike",
        reset = """
            v = v_reset
            w += b
        """,
        refractory = 0.1
    )
    ```

    
    :param v_rest:  Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Decay time of excitatory synaptic current (ms)
    :param tau_syn_I: Decay time of inhibitory synaptic current (ms)
    :param e_rev_E: Reversal potential for excitatory input (mV)
    :param e_rev_I: Reversal potential for inhibitory input (mv)
    :param tau_w: Time constant of the adaptation variable (ms)
    :param a: Scaling of the adaptation variable
    :param b: Increment on the adaptation variable after a spike
    :param i_offset: Offset current (nA)
    :param delta_T: Speed of the exponential (mV)
    :param v_thresh: Spike threshold for the exponential (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param v_spike: Spike threshold (mV)
    '''

    # For reporting
    _instantiated = []
    
    def __init__(self, v_rest = -70.6, cm = 0.281, tau_m = 9.3667, 
        tau_refrac = 0.1, tau_syn_E = 5.0, tau_syn_I = 5.0, 
        e_rev_E = 0.0, e_rev_I = -80.0, tau_w = 144.0, 
        a = 4.0, b = 0.0805, i_offset = 0.0, delta_T = 2.0, 
        v_thresh = -50.4, v_reset = -70.6, v_spike = -40.0):
        
        # Create the arguments
        parameters = """
            v_rest     = %(v_rest)s
            cm         = %(cm)s
            tau_m      = %(tau_m)s
            tau_refrac = %(tau_refrac)s
            tau_syn_E  = %(tau_syn_E)s
            tau_syn_I  = %(tau_syn_I)s
            e_rev_E    = %(e_rev_E)s
            e_rev_I    = %(e_rev_I)s
            tau_w      = %(tau_w)s
            a          = %(a)s
            b          = %(b)s
            i_offset   = %(i_offset)s
            delta_T    = %(delta_T)s
            v_thresh   = %(v_thresh)s
            v_reset    = %(v_reset)s
            v_spike    = %(v_spike)s
        """ % {
            'v_rest'     : v_rest    ,
            'cm'         : cm        ,
            'tau_m'      : tau_m     ,
            'tau_refrac' : tau_refrac,
            'tau_syn_E'  : tau_syn_E ,
            'tau_syn_I'  : tau_syn_I ,
            'e_rev_E'    : e_rev_E   ,
            'e_rev_I'    : e_rev_I   ,
            'tau_w'      : tau_w     ,
            'a'          : a         ,
            'b'          : b         ,
            'i_offset'   : i_offset  ,
            'delta_T'    : delta_T   ,
            'v_thresh'   : v_thresh  ,
            'v_reset'    : v_reset   ,
            'v_spike'    : v_spike   ,
        }
        # Equations for the variables
        equations="""    
            I = g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset

            tau_m * dv/dt = (v_rest - v +  delta_T * exp((v-v_thresh)/delta_T)) + tau_m/cm*(I - w) : init=%(v_reset)s

            tau_w * dw/dt = a * (v - v_rest) / 1000.0 - w 

            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
        """ % {'v_reset': v_reset}

        spike = """
            v > v_spike
        """
        reset = """
            v = v_reset
            w += b
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset, refractory='tau_refrac',
            name="Adaptive exponential Integrate-and-Fire", 
            description="Exponential integrate-and-fire neuron with spike triggered and sub-threshold adaptation currents (isfa, ista reps.).")

        # For reporting
        self._instantiated.append(True)

class EIF_cond_alpha_isfa_ista(Neuron):
    ''' 
    Exponential integrate-and-fire neuron with spike triggered and sub-threshold adaptation conductances (isfa, ista reps.), alpha post-synaptic conductances.
     
    Definition according to:

    > Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    Equivalent code:

    ```python

    EIF_cond_alpha_isfa_ista = Neuron(
        parameters = dict(
            v_rest = ann.Parameter(-70.6),
            cm = ann.Parameter(0.281), 
            tau_m = ann.Parameter(9.3667), 
            tau_syn_E = ann.Parameter(5.0),
            tau_syn_I = ann.Parameter(5.0), 
            e_rev_E = ann.Parameter(0.0),
            e_rev_I = ann.Parameter(-80.0),
            tau_w = ann.Parameter(144.0), 
            a = ann.Parameter(4.0),
            b = ann.Parameter(0.0805),
            i_offset = ann.Parameter(0.0),
            delta_T = ann.Parameter(2.0),
            v_thresh = ann.Parameter(-50.4),
            v_reset = ann.Parameter(-70.6),
            v_spike = ann.Parameter(-40.0),
        ), 
        equations = [
            # Scaling
            'gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)',
            'gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)',

            # Input current
            'I = alpha_exc * (e_rev_E - v) + alpha_inh * (e_rev_I - v) + i_offset',    

            # Membrane potential     
            ann.Variable('tau_m * dv/dt = (v_rest - v +  delta_T * exp((v-v_thresh)/delta_T)) + tau_m/cm*(I - w)', init=-70.6),   

            # Recovery variable 
            'tau_w * dw/dt = a * (v - v_rest) / 1000.0 - w',    

            # Alpha-shaped conductance
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method='exponential'),
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method='exponential'),

            ann.Variable('tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc', method='exponential'),
            ann.Variable('tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh', method='exponential'),
        ],
        spike = "v > v_spike",
        reset = """
            v = v_reset
            w += b
        """,
        refractory = 0.1
    )
    ```

    :param v_rest:  Resting membrane potential (mV)
    :param cm: Capacity of the membrane (nF)
    :param tau_m: Membrane time constant (ms)
    :param tau_refrac: Duration of refractory period (ms)
    :param tau_syn_E: Decay time of excitatory synaptic current (ms)
    :param tau_syn_I: Decay time of inhibitory synaptic current (ms)
    :param e_rev_E: Reversal potential for excitatory input (mV)
    :param e_rev_I: Reversal potential for inhibitory input (mv)
    :param tau_w: Time constant of the adaptation variable (ms)
    :param a: Scaling of the adaptation variable
    :param b: Increment on the adaptation variable after a spike
    :param i_offset: Offset current (nA)
    :param delta_T: Speed of the exponential (mV)
    :param v_thresh: Spike threshold for the exponential (mV)
    :param v_reset: Reset potential after a spike (mV)
    :param v_spike: Spike threshold (mV)
    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, v_rest = -70.6, cm = 0.281, tau_m = 9.3667, 
        tau_refrac = 0.1, tau_syn_E = 5.0, tau_syn_I = 5.0, 
        e_rev_E = 0.0, e_rev_I = -80.0, tau_w = 144.0, 
        a = 4.0, b = 0.0805, i_offset = 0.0, delta_T = 2.0, 
        v_thresh = -50.4, v_reset = -70.6, v_spike = -40.0):
        
        # Create the arguments
        parameters = """
            v_rest     = %(v_rest)s
            cm         = %(cm)s
            tau_m      = %(tau_m)s
            tau_refrac = %(tau_refrac)s
            tau_syn_E  = %(tau_syn_E)s
            tau_syn_I  = %(tau_syn_I)s
            e_rev_E    = %(e_rev_E)s
            e_rev_I    = %(e_rev_I)s
            tau_w      = %(tau_w)s
            a          = %(a)s
            b          = %(b)s
            i_offset   = %(i_offset)s
            delta_T    = %(delta_T)s
            v_thresh   = %(v_thresh)s
            v_reset    = %(v_reset)s
            v_spike    = %(v_spike)s
        """ % {
            'v_rest'     : v_rest    ,
            'cm'         : cm        ,
            'tau_m'      : tau_m     ,
            'tau_refrac' : tau_refrac,
            'tau_syn_E'  : tau_syn_E ,
            'tau_syn_I'  : tau_syn_I ,
            'e_rev_E'    : e_rev_E   ,
            'e_rev_I'    : e_rev_I   ,
            'tau_w'      : tau_w     ,
            'a'          : a         ,
            'b'          : b         ,
            'i_offset'   : i_offset  ,
            'delta_T'    : delta_T   ,
            'v_thresh'   : v_thresh  ,
            'v_reset'    : v_reset   ,
            'v_spike'    : v_spike   ,
        }
        # Equations for the variables
        equations="""    

            gmax_exc = exp((tau_syn_E - dt/2.0)/tau_syn_E)
            gmax_inh = exp((tau_syn_I - dt/2.0)/tau_syn_I)
            
            I = alpha_exc * (e_rev_E - v) + alpha_inh * (e_rev_I - v) + i_offset

            tau_m * dv/dt = (v_rest - v +  delta_T * exp((v-v_thresh)/delta_T)) + tau_m/cm*(I - w) : init=%(v_reset)s

            tau_w * dw/dt = a * (v - v_rest) / 1000.0 - w 

            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
            tau_syn_E * dalpha_exc/dt = gmax_exc * g_exc - alpha_exc  : exponential
            tau_syn_I * dalpha_inh/dt = gmax_inh * g_inh - alpha_inh  : exponential
        """ % {'v_reset': v_reset}

        spike = """
            v > v_spike
        """
        reset = """
            v = v_reset
            w += b
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset, refractory='tau_refrac',
            name="Adaptive exponential Integrate-and-Fire", 
            description="Exponential integrate-and-fire neuron with spike triggered and sub-threshold adaptation conductances (isfa, ista reps.).")

        # For reporting
        self._instantiated.append(True)

##################
### HH
##################
class HH_cond_exp(Neuron):
    '''
    Single-compartment Hodgkin-Huxley-type neuron with transient sodium and delayed-rectifier potassium currents using the ion channel models from Traub.

    The ODEs for n, m, h and v are solved using the midpoint method, while the conductances g_exc and g_inh are solved using the exponential Euler method.

    Equivalent code:

    ```python
    HH_cond_exp = Neuron(
        parameters = dict(
            gbar_Na = 20.0
            gbar_K = 6.0
            gleak = 0.01
            cm = 0.2 
            v_offset = -63.0 
            e_rev_Na = 50.0
            e_rev_K = -90.0 
            e_rev_leak = -65.0
            e_rev_E = 0.0
            e_rev_I = -80.0 
            tau_syn_E = 0.2
            tau_syn_I = 2.0
            i_offset = 0.0
            v_thresh = 0.0
        ), 
        equations = [
            # Previous membrane potential
            'prev_v = v',

            # Voltage-dependent rate constants
            'an = 0.032 * (15.0 - v + v_offset) / (exp((15.0 - v + v_offset)/5.0) - 1.0)',
            'am = 0.32  * (13.0 - v + v_offset) / (exp((13.0 - v + v_offset)/4.0) - 1.0)',
            'ah = 0.128 * exp((17.0 - v + v_offset)/18.0)',

            'bn = 0.5   * exp ((10.0 - v + v_offset)/40.0)',
            'bm = 0.28  * (v - v_offset - 40.0) / (exp((v - v_offset - 40.0)/5.0) - 1.0)',
            'bh = 4.0/(1.0 + exp (( 10.0 - v + v_offset )) )',

            # Activation variables
            ann.Variable('dn/dt = an * (1.0 - n) - bn * n', init = 0.0, method="exponential")
            ann.Variable('dm/dt = am * (1.0 - m) - bm * m', init = 0.0, method="exponential")
            ann.Variable('dh/dt = ah * (1.0 - h) - bh * h', init = 1.0, method="exponential")

            # Membrane equation
            ann.Variable('cm * dv/dt = gleak*(e_rev_leak -v) + gbar_K * n**4 * (e_rev_K - v) + gbar_Na * m**3 * h * (e_rev_Na - v)
                            + g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset', method="exponential", init=-65.0),

            # Exponentially-decaying conductances
            ann.Variable('tau_syn_E * dg_exc/dt = - g_exc', method="exponential")
            ann.Variable('tau_syn_I * dg_inh/dt = - g_inh', method="exponential")
        ],
        spike = "(v > v_thresh) and (prev_v <= v_thresh)",
        reset = ""
    )
    ```

    :param gbar_Na: Maximal conductance of the Sodium current.
    :param gbar_K: Maximal conductance of the Potassium current. 
    :param gleak: Conductance of the leak current (nF)  
    :param cm: Capacity of the membrane (nF)
    :param v_offset:  Threshold for the rate constants (mV)  
    :param e_rev_Na: Reversal potential for the Sodium current (mV) 
    :param e_rev_K: Reversal potential for the Potassium current (mV)  
    :param e_rev_leak: Reversal potential for the leak current (mV)   
    :param e_rev_E: Reversal potential for excitatory input (mV)  
    :param e_rev_I: Reversal potential for inhibitory input (mV)  
    :param tau_syn_E: Decay time of excitatory synaptic current (ms)  
    :param tau_syn_I: Decay time of inhibitory synaptic current (ms)   
    :param i_offset: Offset current (nA)
    :param v_thresh: Threshold for spike emission

    '''
    # For reporting
    _instantiated = []
    
    def __init__(self, gbar_Na = 20.0, gbar_K = 6.0, gleak = 0.01, cm = 0.2, 
        v_offset = -63.0, e_rev_Na = 50.0, e_rev_K = -90.0, e_rev_leak = -65.0, 
        e_rev_E = 0.0, e_rev_I = -80.0, tau_syn_E = 0.2, tau_syn_I = 2.0, 
        i_offset = 0.0, v_thresh = 0.0):

        parameters = """
        gbar_Na    = %(gbar_Na)s   
        gbar_K     = %(gbar_K)s     
        gleak      = %(gleak)s      
        cm         = %(cm)s        
        v_offset   = %(v_offset)s 
        e_rev_Na   = %(e_rev_Na)s 
        e_rev_K    = %(e_rev_K)s   
        e_rev_leak = %(e_rev_leak)s   
        e_rev_E    = %(e_rev_E)s    
        e_rev_I    = %(e_rev_I)s   
        tau_syn_E  = %(tau_syn_E)s 
        tau_syn_I  = %(tau_syn_I)s 
        i_offset   = %(i_offset)s  
        v_thresh   = %(v_thresh)s  
        """ % {
        'gbar_Na'    : gbar_Na   ,
        'gbar_K'     : gbar_K     ,
        'gleak'      : gleak      ,
        'cm'         : cm        ,
        'v_offset'   : v_offset  ,
        'e_rev_Na'   : e_rev_Na  ,
        'e_rev_K'    : e_rev_K   ,
        'e_rev_leak' : e_rev_leak   ,
        'e_rev_E'    : e_rev_E    ,
        'e_rev_I'    : e_rev_I   ,
        'tau_syn_E'  : tau_syn_E ,
        'tau_syn_I'  : tau_syn_I ,
        'i_offset'   : i_offset  ,
        'v_thresh'   : v_thresh  
        }

        equations = """
            # Previous membrane potential
            prev_v = v

            # Voltage-dependent rate constants
            an = 0.032 * (15.0 - v + v_offset) / (exp((15.0 - v + v_offset)/5.0) - 1.0)
            am = 0.32  * (13.0 - v + v_offset) / (exp((13.0 - v + v_offset)/4.0) - 1.0)
            ah = 0.128 * exp((17.0 - v + v_offset)/18.0) 

            bn = 0.5   * exp ((10.0 - v + v_offset)/40.0)
            bm = 0.28  * (v - v_offset - 40.0) / (exp((v - v_offset - 40.0)/5.0) - 1.0)
            bh = 4.0/(1.0 + exp (( 10.0 - v + v_offset )) )

            # Activation variables
            dn/dt = an * (1.0 - n) - bn * n : init = 0.0, exponential
            dm/dt = am * (1.0 - m) - bm * m : init = 0.0, exponential
            dh/dt = ah * (1.0 - h) - bh * h : init = 1.0, exponential

            # Membrane equation
            cm * dv/dt = gleak*(e_rev_leak -v) + gbar_K * n**4 * (e_rev_K - v) + gbar_Na * m**3 * h * (e_rev_Na - v) 
                + g_exc * (e_rev_E - v) + g_inh * (e_rev_I - v) + i_offset: exponential, init=%(e_rev_leak)s

            # Exponentially-decaying conductances
            tau_syn_E * dg_exc/dt = - g_exc : exponential
            tau_syn_I * dg_inh/dt = - g_inh : exponential
        """ % {'e_rev_leak': e_rev_leak}

        spike = "(v > v_thresh) and (prev_v <= v_thresh)"

        reset = ""

        Neuron.__init__(self, parameters=parameters, equations=equations, 
            spike=spike, reset=reset,
            name="Hodgkin-Huxley", 
            description="Single-compartment Hodgkin-Huxley-type neuron with transient sodium and delayed-rectifier potassium currents.")

        # For reporting
        self._instantiated.append(True)