#===============================================================================
#
#     InputEncoding.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022    Abdul Rehaman Kampli <>
#                           René Larisch <>
#                           Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
from ANNarchy.core.Neuron import Neuron

#=====================================================ENCODING TECHNICHS =========================================================================
#
#	The Intrinsically Bursting and Chattering type of encoding has implemented by referring to the paper "Simple Model of Spiking Neurons"
#	by Eugene M. Izhikevich, this is achieved by manipulating the parameters of Izhikevich Neuron and the nature of the neuron activity
#	of the these two technichs are as of Burst Coding.
#
#	The Phase Coding Oscillation type of encoidng has implemented by referring ot the paper "Deep neural networks with weighted spikes"
#	by Jaehyun Kim et al., this is achieved by writing the equation of the phase(equation 6) of the referred paper and multiplying phase
#	with the thershold(vt being constant) value to set the new dynamic thershold.
#
#   TODO: Documentation for IF
#

__all__ = ["IB", "PSO", "CH"]

#====================================================Intrinsically Bursting========================================================================

IB = Neuron(
    parameters = """
        a = 0.02
        b = 0.2
        c = -55.0
        d = 4.0
        v_thresh = 30.0
        rates=0.0
    """,
    equations = """
        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + rates : init = -55.0
        du/dt = a * (b*v - u) : init= -13.0
    """,
    spike = "v > v_thresh",
    reset = "v = c; u += d",
    refractory = 0.0
)

#======================================================Chattering====================================================================================

CH = Neuron(
    parameters = """
        a = 0.02
        b = 0.2
        c = -50.0
        d = 2.0
        v_thresh = 30.0
        rates=0.0
    """,
    equations = """
        dv/dt = 0.04 * v^2 + 5.0 * v + 140.0 - u + rates : init = -50.0
        du/dt = a * (b*v - u) : init= -13.0
    """,
    spike = "v > v_thresh",
    reset = "v = c; u += d",
    refractory = 0.0
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
    """,
    spike = """
        v > vt_new
    """
)

#=====================================================================================================================================================
