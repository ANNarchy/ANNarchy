"""

    storage_formats.py

    This file is part of ANNarchy.

    Copyright (C) 2022 Alex Schwarz and Helge Uelo Dinkelbach
    <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

# Defines the tested storage_formats. The tests are overloaded in __init__.py.
# In most cases, we have the rule that filename is equal to test case. Otherwise,
# the corresponding file is added as comment.

single_thread = {
    # test_SpikingContinuousUpdate.py
    "test_SpikingContinuousUpdate":             ["lil", "csr"],
    "test_ContinuousTransmission":              ["lil", "csr"],
    # from test_SpikingSynapse
    "test_PreSpike":                            ["lil", "csr"],
    "test_PostSpike":                           ["lil", "csr"],
    "test_TimeDependentUpdate":                 ["lil", "csr"],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay":            ["lil", "csr", "dense"],
    "test_SpikeTransmissionUniformDelay":       ["lil", "csr"],
    "test_SpikeTransmissionNonUniformDelay":    ["lil"],
}

open_mp = {
    # from test_ContinuousUpdate.py
    "test_SpikingContinuousUpdate":             ["lil", "csr"],
    # from test_SpikingSynapse
    "test_PreSpike":                            ["lil", "csr"],
    "test_PostSpike":                           ["lil", "csr"],
    "test_TimeDependentUpdate":                 ["lil", "csr"],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay":            ["lil", "csr"],
    "test_SpikeTransmissionUniformDelay":       ["lil", "csr"],
    "test_SpikeTransmissionNonUniformDelay":    ["lil"],
}

cuda = {
    # from test_SpikingContinuousUpdate.py
    "test_SpikingContinuousUpdate":         ["csr"],
    # from test_SpikingSynapse
    "test_PreSpike":                        ["csr", ],
    "test_PostSpike":                       ["csr", ],
    # from test_SpikingTransmission
    "test_SpikeTransmissionNoDelay":        ["csr"],
    "test_SpikeTransmissionUniformDelay":   ["csr"],
}
