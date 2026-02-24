"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Defines the tested storage_formats. The tests are overloaded in __init__.py.
# In most cases, we have the rule that filename is equal to test case. Otherwise,
# the corresponding file is added as comment.

single_thread = {
    # test_SpikingContinuousUpdate.py
    "test_SpikingContinuousUpdate": ["lil", "csr"],
    "test_ContinuousTransmission": ["lil", "csr"],
    # from test_SpikingSynapse
    "test_PreSpike": ["lil", "csr", "dense"],
    "test_PostSpike": ["lil", "csr", "dense"],
    "test_TimeDependentUpdate": ["lil", "csr", "dense"],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay": ["lil", "csr", "dense"],
    "test_SpikeTransmissionUniformDelay": ["lil", "csr"],
    "test_SpikeTransmissionNonUniformDelay": ["lil"],
    # from test_SpikingDefaultSynapses
    "test_SpikingDefaultSynapses": ["lil", "csr"],
}

open_mp = {
    # from test_ContinuousUpdate.py
    "test_SpikingContinuousUpdate": ["lil", "csr"],
    "test_ContinuousTransmission": ["lil", "csr"],
    # from test_SpikingSynapse
    "test_PreSpike": ["lil", "csr", "dense"],
    "test_PostSpike": ["lil", "csr", "dense"],
    "test_TimeDependentUpdate": ["lil", "csr", "dense"],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay": ["lil", "csr", "dense"],
    "test_SpikeTransmissionUniformDelay": ["lil", "csr"],
    "test_SpikeTransmissionNonUniformDelay": ["lil"],
    # from test_SpikingDefaultSynapses
    "test_SpikingDefaultSynapses": ["lil", "csr"],
}

cuda = {
    # from test_SpikingContinuousUpdate.py
    "test_SpikingContinuousUpdate": ["csr"],
    "test_ContinuousTransmission": ["csr"],
    # from test_SpikingSynapse
    "test_PreSpike": [
        "csr",
    ],
    "test_PostSpike": [
        "csr",
    ],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay": ["csr"],
    "test_SpikeTransmissionUniformDelay": ["csr"],
    # from test_SpikingDefaultSynapses
    "test_SpikingDefaultSynapses": ["csr"],
}
