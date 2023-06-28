#===============================================================================
#
#     ReadOut.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022-23 Abdul Rehaman Kampli <>
#                           René Larisch <renelarischif@gmail.com>
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

# Default neuron used in hidden layers and output layer
IaF = Neuron(
    parameters = """
        vt = 1          : population
        vr = 0          : population
        mask_tau = 20.0 : population
    """,
    equations = """
        dv/dt    = g_exc          : init = 0.0 , min=-2.0
        dmask/dt = -mask/mask_tau : init = 0.0
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
        mask += 1/mask_tau
    """
)

# Used as output layer for "time_to_k_spikes" read-out.
IaF_TTKS = Neuron(
    parameters = """
        k = 0           : population
        vt = 1          : population
        vr = 0          : population
        sc = 0
    """,
    equations = """
        dv/dt = g_exc   : init = 0.0 , min=-2.0
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
        sc += 1
    """
)

# Used as output layer for "max_membrane_potential" read-out.
IaF_Acc = Neuron(
    equations = """
        dv/dt = g_exc   : init = 0.0 , min=-2.0
    """,
    spike = """
        v < -10000 # will never happen
    """,
    reset = ""
)

available_read_outs = [
    "spike_count",
    "time_to_first_spike",
    "time_to_k_spikes",
    "membrane_potential"
]