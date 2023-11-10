"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Neuron import Neuron

available_read_outs = [
    "spike_count",
    "time_to_first_spike",
    "time_to_k_spikes",
    "membrane_potential"
]

# Default neuron used in hidden layers
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
        mask += 1.0/mask_tau
    """,
    name = "IaF neuron"
)

# Default neuron used in read-out layer
IaF_ReadOut = Neuron(
    parameters = """
        vt = 1          : population
        vr = 0          : population
    """,
    equations = """
        dv/dt = g_exc   : init = 0.0 , min=-2.0
    """,
    spike = """
        v > vt
    """,
    reset = """
        v = vr
    """,
    name = "IaF neuron"
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
