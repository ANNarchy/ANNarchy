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
single_thread = {
    "test_RateTransmission":    ["lil", "csr", "ell"],
    "test_CustomConnectivity":  ["lil", "csr", "ell"],
    "test_Dendrite":            ["lil", "csr", "ell"],
    "test_Projection":          ["lil", "csr", "ell"],
    # "test_Convolution":         ["lil", "csr", "ell"],
    # "test_Pooling":             ["lil", "csr", "ell"],
    "test_Locality":            ["lil", "csr", "ell"],  # ell?
    "test_AccessPSP":           ["lil", "csr", "ell"],
    "test_ModifiedPSP":         ["lil", "csr", "ell"],
    "test_NoDelay":             ["lil", "csr", "ell"],
    "test_UniformDelay":        ["lil", "csr", "ell"], # csr/ell sollten funkt.
    "test_ContinuousUpdate":    ["lil", "csr", "ell"],
    "test_SynapticAccess":      ["lil", "csr", "ell"],
    "test_PreSpike":            ["lil", "csr", ],
    "test_PostSpike":           ["lil", "csr", ],
    "test_SpikeTransmission":   ["lil", "csr", ],
    "test_CurrentInjection":    ["lil", "csr", ],
    "test_SynapseOperations":   ["lil", ],
}

p2p = ["test_Dendrite", "test_Projection", "test_PreSpike", "test_PostSpike",
       "test_CurrentInjection"]
