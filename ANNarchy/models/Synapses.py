"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Synapse import Synapse
from ANNarchy.intern import Messages


def list_standard_synapses():
    "Returns a list of standard neuron models available."
    return [STP, STDP, Hebb, Oja, IBCM]


###############################
### Default rate-coded Synapse
###############################
class DefaultRateCodedSynapse(Synapse):
    # For reporting
    _instantiated = []
    def __init__(self):
        Synapse.__init__(self, 
            equations="",
            psp="w * pre.r",
            name="-", 
            description="Weighted sum of firing rates."
        )
        # For reporting
        self._instantiated.append(True)

###############################
### Default spiking Synapse
###############################
class DefaultSpikingSynapse(Synapse):
    # For reporting
    _instantiated = []
    def __init__(self):
        Synapse.__init__(self, 
            pre_spike = "g_target += w",
            name="Event-driven synapse", 
            description="Increases the post-synaptic conductance from the synaptic efficency after each pre-synaptic spike."
        )
        # For reporting
        self._instantiated.append(True)

##################
### Hebb
##################
class Hebb(Synapse):
    '''
    Rate-coded synapse with Hebbian plasticity.

    Equivalent code:

    ```python
    Hebb = ann.Synapse(
        parameters = dict(
            eta = 0.01,
        ),
        equations = [
            ann.Variable('dw/dt = eta * pre.r * post.r', min=0.0),
        ]
    )
    ```

    :param eta: learning rate.
    '''
    # For reporting
    _instantiated = []

    def __init__(self, eta=0.01):

        parameters = """
            eta = %(eta)s : projection
        """ % {'eta': eta}

        equations = """
            dw/dt = eta * pre.r * post.r : min=0.0, explicit 
        """

        Synapse.__init__(
            self, 
            parameters=parameters, 
            equations=equations,
            name="Hebbian Plasticity", 
            description="Simple Hebbian learning rule"
        )
        
        # For reporting
        self._instantiated.append(True)

##################
### Oja
##################
class Oja(Synapse):
    '''
    Rate-coded synapse with regularized Hebbian plasticity (Oja).
    
    Equivalent code:

    ```python
    Oja = ann.Synapse(
        parameters = dict(
            eta = 0.01,
            alpha = 1.0,
        ),
        equations = [
            ann.Variable('dw/dt = eta * ( pre.r * post.r - alpha * post.r^2 * w )', min=0.0),
        ]
    )
    ```

    :param eta: learning rate.
    :param alpha: regularization coefficient.
    '''
    # For reporting
    _instantiated = []

    def __init__(self, eta:float=0.01, alpha:float=1.0):

        parameters = """
            eta = %(eta)s : projection
            alpha = %(alpha)s : projection
        """ % {'eta': eta, 'alpha': alpha}

        equations = """
            dw/dt = eta * ( pre.r * post.r - alpha * post.r^2 * w ) : min=0.0, explicit 
        """

        Synapse.__init__(self, parameters=parameters, equations=equations,
            name="Oja plasticity", description="Regularized Hebbian learning rule.")
        # For reporting
        self._instantiated.append(True)

##################
### IBCM
##################
class IBCM(Synapse):
    '''
    Rate-coded synapse with Intrator & Cooper (1992) plasticity.

    Equivalent code:

    ```python
    IBCM = ann.Synapse(
        parameters = dict(
            eta = 0.01,
            tau = 2000.0,
        ),
        equations = [
            ann.Variable('tau * dtheta/dt + theta = post.r^2', locality='semiglobal', method='exponential'),
            ann.Variable('dw/dt = eta * post.r * (post.r - theta) * pre.r',' min=0.0),
        ]
    )
    ```

    :param eta: learning rate.
    :param tau: time constant of the sliding threshold.
    '''
    # For reporting
    _instantiated = []

    def __init__(self, eta = 0.01, tau = 2000.0):

        parameters = """
            eta = %(eta)s : projection
            tau = %(tau)s : projection
        """ % {'eta': eta, 'tau': tau}

        equations = """
            tau * dtheta/dt + theta = post.r^2 : postsynaptic, exponential
            dw/dt = eta * post.r * (post.r - theta) * pre.r : min=0.0, explicit
        """

        Synapse.__init__(self, parameters=parameters, equations=equations,
            name="IBCM", description="Intrator and Cooper (1992) learning rule.")
        # For reporting
        self._instantiated.append(True)

##################
### STP
##################
class STP(Synapse):
    '''
    Synapse exhibiting short-term facilitation and depression.
    
    Implemented using the model of Tsodyks, Markram et al.:

    > Tsodyks, Uziel and Markram (2000) Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses. Journal of Neuroscience 20:RC50

    Note that the time constant of the post-synaptic current is set in the neuron model, not here.
    
    Equivalent code:

    ```python
    STP = ann.Synapse(
        parameters = dict(
            tau_rec = 100.0,
            tau_facil = 0.01,
            U = 0.5,
        ),
        equations = [
            ann.Variable('dx/dt = (1 - x)/tau_rec', init = 1.0, method='event-driven'),
            ann.Variable('du/dt = (U - u)/tau_facil', init = 0.5, method='event-driven'),
        ],
        pre_spike="""
            g_target += w * u * x
            x *= (1 - u)
            u += U * (1 - u)
        """
    )
    ```

    :param tau_rec: depression time constant (ms).
    :param tau_facil: facilitation time constant (ms).
    :param U: use parameter.
    '''
    # For reporting
    _instantiated = []

    def __init__(self, tau_rec=100.0, tau_facil=0.01, U=0.5):

        if tau_facil<= 0.0:
            Messages._error('STP: tau_facil must be positive. Choose a very small value if you have to, or derive a new synapse.')
            
        parameters = """
            tau_rec = %(tau_rec)s : projection
            tau_facil = %(tau_facil)s : projection
            U = %(U)s
        """ % {'tau_rec': tau_rec, 'tau_facil': tau_facil, 'U': U}
        equations = """
            dx/dt = (1 - x)/tau_rec : init = 1.0, event-driven
            du/dt = (U - u)/tau_facil : init = %(U)s, event-driven   
        """ % {'tau_rec': tau_rec, 'tau_facil': tau_facil, 'U': U}
        pre_spike="""
            g_target += w * u * x
            x *= (1 - u)
            u += U * (1 - u)
        """

        Synapse.__init__(self, parameters=parameters, equations=equations, pre_spike=pre_spike,
            name="Short-term plasticity", description="Synapse exhibiting short-term facilitation and depression, implemented using the model of Tsodyks, Markram et al.")
        # For reporting
        self._instantiated.append(True)
        

##################
### STDP
##################
class STDP(Synapse):
    '''
    Spike-timing dependent plasticity, online version.

    > Song, S., and Abbott, L.F. (2001). Cortical development and remapping through spike timing-dependent plasticity. Neuron 32, 339-350. 
    
    Equivalent code:

    ```python
    STDP = ann.Synapse(
        parameters = dict(
            tau_plus = 20.0,
            tau_minus = 20.0,
            A_plus = 0.01,
            A_minus = 0.01,
            w_min = 0.0,
            w_max = 1.0,
        ),
        equations = [
            ann.Variable('tau_plus  * dx/dt = -x', method='event-driven'),
            ann.Variable('tau_minus * dy/dt = -y', method='event-driven'),
        ],
        pre_spike="""
            g_target += w
            x += A_plus * w_max
            w = clip(w + y, w_min , w_max)
        """,
        post_spike="""
            y -= A_minus * w_max
            w = clip(w + x, w_min , w_max)
        """
    )
    ```

    
    :param tau_plus: time constant of the pre-synaptic trace (ms)
    :param tau_minus: time constant of the pre-synaptic trace (ms)
    :param A_plus: increase of the pre-synaptic trace after a spike.
    :param A_minus: decrease of the post-synaptic trace after a spike. 
    :param w_min: minimal value of the weight w.
    :param w_max: maximal value of the weight w.
    '''
    # For reporting
    _instantiated = []

    def __init__(self, tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.01, w_min=0.0, w_max=1.0):

        parameters="""
            tau_plus = %(tau_plus)s : projection
            tau_minus = %(tau_minus)s : projection
            A_plus = %(A_plus)s : projection
            A_minus = %(A_minus)s : projection
            w_min = %(w_min)s : projection
            w_max = %(w_max)s : projection
        """ % {'tau_plus': tau_plus, 'tau_minus':tau_minus, 'A_plus':A_plus, 'A_minus': A_minus, 'w_min': w_min, 'w_max': w_max}

        equations = """
            tau_plus  * dx/dt = -x : event-driven
            tau_minus * dy/dt = -y : event-driven
        """
        pre_spike="""
            g_target += w
            x += A_plus * w_max
            w = clip(w + y, w_min , w_max)
        """          
        post_spike="""
            y -= A_minus * w_max
            w = clip(w + x, w_min , w_max)
        """

        Synapse.__init__(self, parameters=parameters, equations=equations, pre_spike=pre_spike, post_spike=post_spike,
            name="Spike-timing dependent plasticity", description="Synapse exhibiting spike-timing dependent plasticity.")
        # For reporting
        self._instantiated.append(True)


