"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Neuron import Neuron

#=====================================================ENCODING TECHNIQUES =========================================================================
#
#	The Intrinsically Bursting and Chattering type of encoding has implemented by referring to the paper "Simple Model of Spiking Neurons"
#	by Eugene M. Izhikevich, this is achieved by manipulating the parameters of Izhikevich Neuron and the nature of the neuron activity
#	of the these two technichs are as of Burst Coding.
#
#	The Phase Coding Oscillation type of encoidng has implemented by referring ot the paper "Deep neural networks with weighted spikes"
#	by Jaehyun Kim et al., this is achieved by writing the equation of the phase(equation 6) of the referred paper and multiplying phase
#	with the threshold (vt being constant) value to set the new dynamic thershold.
#

__all__ = ["CPN", "IB", "PSO"]

#====================================================Custom Poisson Neuron========================================================================

CPN = Neuron(
    parameters = """
        rates = 0.0
        mask_tau = 20.0
    """,
    equations = """
        p = (Uniform(0.0, 1.0) * 1000.0) / dt
        dmask/dt = -mask/mask_tau : init=0.0
    """,
    spike = "p < rates",
    reset="mask += 1.0/mask_tau",
    name="custom poisson neuron"
)

#====================================================Intrinsically Bursting========================================================================

IB = Neuron(
    parameters = """
        a = 0.02
        b = 0.2
        c = -55.0
        d = 4.0
        v_thresh = 30.0
        rates=0.0
        mask_tau = 20
    """,
    equations = """
        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + rates : init = -55.0
        du/dt = a * (b*v - u) : init= -13.0
        dmask/dt = -mask/mask_tau : init=0.0
    """,
    spike = "v > v_thresh",
    reset = "v = c; u += d; mask += 1.0/mask_tau",
    name = "intrinsically bursting"
)

#=====================================================phase shift oscillation==========================================================================

PSO = Neuron(
    parameters = """
        k= 8
        vt = 1
        rates=0
    """,
    equations = """
        p= pow(2,(-1+(modulo(t-1,k))))
        vt_new=p*vt
        v = rates : init = 0
        dmask/dt = -mask/mask_tau : init=0.0
    """,
    spike = "v > vt_new",
    reset = "mask += 1.0/mask_tau",
    name = "phase shift oscillation"
)

#=====================================================================================================================================================
