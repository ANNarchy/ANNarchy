#===============================================================================
#
#     Synapses.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core.Global import _error


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

    **Parameters (global)**:

    * eta = 0.01 : learning rate.

    **Learning rule**:

    * w : weight.

    ```
    dw/dt = eta * pre.r * post.r
    ```

    Equivalent code:

    ```python
    Hebb = Synapse(
        parameters = """
            eta = 0.01 : projection
        """,
        equations = """
            dw/dt = eta * pre.r * post.r : min=0.0
        """
    )
    ```

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

        Synapse.__init__(self, parameters=parameters, equations=equations,
            name="Hebbian Plasticity", description="Simple Hebbian learning rule")
        # For reporting
        self._instantiated.append(True)

##################
### Oja
##################
class Oja(Synapse):
    '''
    Rate-coded synapse with regularized Hebbian plasticity (Oja).

    **Parameters (global)**:

    * eta = 0.01 : learning rate.

    * alpha = 1.0 : regularization constant.

    **Learning rule**:

    * w : weight:

    ```
    dw/dt = eta * ( pre.r * post.r - alpha * post.r^2 * w )
    ```
    
    Equivalent code:

    ```python
    Oja = Synapse(
        parameters = """
            eta = 0.01 : projection
            alpha = 1.0 : projection
        """,
        equations = """
            dw/dt = eta * ( pre.r * post.r - alpha * post.r^2 * w ) : min=0.0
        """
    )
    ```

    '''
    # For reporting
    _instantiated = []

    def __init__(self, eta=0.01, alpha=1.0):

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

    **Parameters (global)**:

    * eta = 0.01 : learning rate.

    * tau = 2000.0 : time constant of the post-synaptic threshold.

    **Learning rule**:

    * theta : post-synaptic threshold:

    ```
    tau * dtheta/dt + theta = post.r^2
    ```

    * w : weight:

    ```
    dw/dt = eta * post.r * (post.r - theta) * pre.r 
    ```
    
    Equivalent code:

    ```python
    IBCM = Synapse(
        parameters = """
            eta = 0.01 : projection
            tau = 2000.0 : projection
        """,
        equations = """
            tau * dtheta/dt + theta = post.r^2 : postsynaptic, exponential
            dw/dt = eta * post.r * (post.r - theta) * pre.r : min=0.0, explicit
        """
    )
    ```

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
    Synapse exhibiting short-term facilitation and depression, implemented using the model of Tsodyks, Markram et al.:

    Tsodyks, Uziel and Markram (2000) Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses. Journal of Neuroscience 20:RC50

    Note that the time constant of the post-synaptic current is set in the neuron model, not here.

    **Parameters (global)**:

    * tau_rec = 100.0 : depression time constant (ms).
    * tau_facil = 0.01 : facilitation time constant (ms).
    * U = 0.5 : use parameter.

    **Variables**:

    * x : recovery variable::

    ```
    dx/dt = (1 - x)/tau_rec 
    ```

    * u : facilitation variable::

    ```
    du/dt = (U - u)/tau_facil 
    ```

    Both variables are integrated event-driven. 

    **Pre-spike events**:

    ```
    g_target += w * u * x
    x *= (1 - u)
    u += U * (1 - u)
    ```
    
    Equivalent code:

    ```python
    STP = Synapse(
        parameters = """
            tau_rec = 100.0 : projection
            tau_facil = 0.01 : projection
            U = 0.5
        """,
        equations = """
            dx/dt = (1 - x)/tau_rec : init = 1.0, event-driven
            du/dt = (U - u)/tau_facil : init = 0.5, event-driven
        """,
        pre_spike="""
            g_target += w * u * x
            x *= (1 - u)
            u += U * (1 - u)
        """
    )
    ```

    '''
    # For reporting
    _instantiated = []

    def __init__(self, tau_rec=100.0, tau_facil=0.01, U=0.5):

        if tau_facil<= 0.0:
            _error('STP: tau_facil must be positive. Choose a very small value if you have to, or derive a new synapse.')
            
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
    Spike-timing dependent plasticity.

    This is the online version of the STDP rule.

    Song, S., and Abbott, L.F. (2001). Cortical development and remapping through spike timing-dependent plasticity. Neuron 32, 339-350. 

    **Parameters (global)**:

    * tau_plus = 20.0 : time constant of the pre-synaptic trace (ms)
    * tau_minus = 20.0 : time constant of the pre-synaptic trace (ms)
    * A_plus = 0.01 : increase of the pre-synaptic trace after a spike.
    * A_minus = 0.01 : decrease of the post-synaptic trace after a spike. 
    * w_min = 0.0 : minimal value of the weight w.
    * w_max = 1.0 : maximal value of the weight w.

    **Variables**:

    * x : pre-synaptic trace:

    ```
    tau_plus  * dx/dt = -x
    ```

    * y: post-synaptic trace:

    ```
    tau_minus * dy/dt = -y
    ```

    Both variables are evaluated event-driven.

    **Pre-spike events**:

    ```
    g_target += w

    x += A_plus * w_max

    w = clip(w + y, w_min , w_max)
    ```

    **Post-spike events**::

    ```
    y -= A_minus * w_max
    
    w = clip(w + x, w_min , w_max)
    ```
    
    Equivalent code:

    ```python

    STDP = Synapse(
        parameters = """
            tau_plus = 20.0 : projection
            tau_minus = 20.0 : projection
            A_plus = 0.01 : projection
            A_minus = 0.01 : projection
            w_min = 0.0 : projection
            w_max = 1.0 : projection
        """,
        equations = """
            tau_plus  * dx/dt = -x : event-driven
            tau_minus * dy/dt = -y : event-driven
        """,
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


