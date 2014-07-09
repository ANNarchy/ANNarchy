from ANNarchy.core.Neuron import Neuron

##################
### Izhikevich
##################
class IzhikevichClass(Neuron):
    """ 
    Default implementation of the Izhikevich neuron.

    By default, the conductance is "g_exc - g_inh", but this can be changed with the ``conductance`` method::

        neuron = Izhikevich.conductance('g_ampa * (1 + g_nmda) - g_gaba')

    Parameters:

    * a : (default: 0.2)
    * b : (default: 0.2)
    * c : (default: -65.0)
    * d : (default: 2.0)
    * v_thresh : threshold on the membrane potential above which a spike is emitted (default: 30.0)
    * noise: amplitude of the normal additive noise (default: 0.0)
    * Iext: external current (default: 0.0)

    Variables:

    * I : input current (user-defined conductance + external current + normal noise).

        I = conductance + Iext + noise * Normal(0.0, 1.0)

    * v : membrane potential in mV (init=-65.0).

        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I 

    * u : recovery variable (init=0.0).

        du/dt = a * (b*v - u) 

    Spike emission:

        v > v_thresh

    Reset:

        v = c
        u += d : unless_refractory 

    Refractory:

        None by defaut.
    """
    def __init__(self, conductance="g_exc - g_inh"):
        # Extract which targets are defined in the conductance
        import re
        targets = re.findall(r'g_([\w]+)', conductance)
        # Create the arguments
        parameters = """
    noise = 0.0
    a = 0.02
    b = 0.2
    c = -65.0
    d = 2.0 
    v_thresh = 30.0
    Iext = 0.0
"""
        # Equations for the variables
        equations="""
    I = %(conductance)s + noise * Normal(0.0, 1.0) + Iext
    dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + I : init = -65.0
    du/dt = a * (b*v - u) 
""" % { 'conductance' : conductance }

        # Default behavior for the conductances (avoid warning)
        for target in targets:
            equations += """
    g_%(target)s = 0.0""" % {'target' : target}

        print equations

        spike = """
    v >= v_thresh
"""
        reset = """
    v = c
    u += d: unless_refractory
"""
        Neuron.__init__(self, parameters=parameters, equations=equations, spike=spike, reset=reset)

    def conductance(self, conductance):
        """
        Sets the conductance using in the equation of I.

        Default: "g_exc - g_inh"
        """
        return IzhikevichClass(conductance=conductance)

    def __repr__(self):
        return self.__doc__

Izhikevich = IzhikevichClass()

