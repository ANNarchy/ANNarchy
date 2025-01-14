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
    # test_RateTransmission.py
    "test_RateTransmission":                    ["lil", "csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay":           ["lil", "csr", "ell"],
    "test_CustomConnectivityUniformDelay":      ["lil", "csr", "ell"],
    "test_CustomConnectivityNonUniformDelay":   ["lil", "csr", "ell"],
    # test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate":           ["lil", "csr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels":            ["lil", "csr", "dense"],
    # from test_Projection.py
    "test_Projection":                          ["lil", "csr"],
    "test_SliceProjections":                    ["lil", "csr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse":              ["lil", "csr", "ell", "dense"],
    "test_DendriteModifiedSynapse":             ["lil", "csr"],
    # from test_RateSynapse.py
    "test_Locality":                            ["lil", "csr", "dense"],
    "test_AccessPSP":                           ["lil", "csr", "dense"],
    "test_ModifiedPSP":                         ["lil", "csr", "dense"],
    # from test_RateDelays
    "test_NoDelay":                             ["lil", "csr", "ell"],
    "test_UniformDelay":                        ["lil", "csr"],
    "test_NonUniformDelay":                     ["lil", "csr"],
    "test_SynapseOperations":                   ["lil"],
    "test_SynapticAccess":                      ["lil", "csr"],
    # SpecificProjections
    "test_Convolution":         ["lil"],
    "test_Pooling":             ["lil"],
    "test_CurrentInjection":    ["lil"],
}

open_mp = {
    # test_RateTransmission.py
    "test_RateTransmissionOneToOne":            ["csr", "ellr", "sell"],
    "test_RateTransmissionAllToAll":            ["csr", "ellr", "sell", "dense"],
    "test_RateTransmissionFixedNumberPre":      ["csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay":           ["lil", "csr", "ell"],
    "test_CustomConnectivityUniformDelay":      ["lil", "csr", "ell"],
    "test_CustomConnectivityNonUniformDelay":   ["lil", "csr", "ell"],
    # from test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate":           ["lil", "csr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels":            ["lil", "csr", "dense"],
    # from test_Projection.py
    "test_Projection":                          ["lil", "csr"],
    "test_SliceProjections":                    ["lil", "csr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse":              ["lil", "csr", "ell", "dense"],
    "test_DendriteModifiedSynapse":             ["lil", "csr"],
    # from test_RateSynapse.py
    "test_Locality":                            ["lil", "csr", "dense"],
    "test_AccessPSP":                           ["lil", "csr", "dense"],
    "test_ModifiedPSP":                         ["lil", "csr", "dense"],
    # from test_RateDelays
    "test_NoDelay":                             ["lil", "csr", "ell"],
    "test_UniformDelay":                        ["lil", "csr"],
    "test_NonUniformDelay":                     ["lil", "csr"],
    "test_SynapseOperations":                   ["lil"],
    "test_SynapticAccess":                      ["lil", "csr"],
    # SpecificProjections
    "test_Convolution":         ["lil"],
    "test_Pooling":             ["lil"],
    "test_CurrentInjection":    ["lil"],
}

cuda = {
    # test_RateTransmission.py
    "test_RateTransmissionOneToOne":        ["csr", "ellr", "sell"],
    "test_RateTransmissionAllToAll":        ["csr", "ellr", "sell"],
    "test_RateTransmissionFixedNumberPre":  ["csr", "ellr", "sell"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay":       ["csr", "ellr"],
    "test_CustomConnectivityUniformDelay":  ["csr", "ellr"],
    # from test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate":       ["csr", "ellr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels":        ["csr", "ellr"],
    # from test_Projection.py
    "test_Projection":                      ["csr", "ellr"],
    "test_SliceProjections":                ["csr", "ellr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse":          ["csr", "ellr"],
    "test_DendriteModifiedSynapse":         ["csr"],
    # from test_RateSynapse.py
    "test_Locality":                        ["csr", "ellr", "dense"],
    "test_AccessPSP":                       ["csr", "ellr", "dense"],
    "test_ModifiedPSP":                     ["csr", "ellr", "dense"],
    # from test_RateDelays
    "test_NoDelay":                         ["csr", "ellr"],
    "test_UniformDelay":                    ["csr", "ellr"],
    # "test_NonUniformDelay":                 ["csr", ],
    "test_SynapseOperations":               ["csr", ],
    "test_SynapticAccess":                  ["csr", "ellr"],
    # SpecificProjections
    "test_Convolution":         ["csr"],
    "test_Pooling":             ["csr"],
    "test_CurrentInjection":    ["csr"],
}
