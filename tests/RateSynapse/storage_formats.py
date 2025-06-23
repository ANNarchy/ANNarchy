"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Defines the tested storage_formats. The tests are overloaded in __init__.py.
# In most cases, we have the rule that filename is equal to test case. Otherwise,
# the corresponding file is added as comment.

single_thread = {
    # test_RateTransmission.py
    "test_RateTransmission":                    ["lil", "csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay":           ["lil", "csr", "ell", "dense"],
    "test_CustomConnectivityUniformDelay":      ["lil", "csr", "ell", "dense"],
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
    "test_NoDelay":                             ["lil", "csr", "ell", "dense"],
    "test_UniformDelay":                        ["lil", "csr", "ell", "dense"],
    "test_NonUniformDelay":                     ["lil", "csr"],
    "test_SynapseOperations":                   ["lil"],
    "test_SynapticAccess":                      ["lil", "csr"],
    # test_Monitor
    "test_MonitorRatePSP":                      ["lil"],
    "test_MonitorLocalVariable":                ["lil"],
    # SpecificProjections
    "test_Convolution":                         ["lil"],
    "test_Pooling":                             ["lil"],
    "test_CurrentInjection":                    ["lil"],
}

open_mp = {
    # test_RateTransmission.py
    "test_RateTransmission":                    ["lil", "csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay":           ["lil", "csr", "ell", "dense"],
    "test_CustomConnectivityUniformDelay":      ["lil", "csr", "ell", "dense"],
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
    "test_NoDelay":                             ["lil", "csr", "ell", "dense"],
    "test_UniformDelay":                        ["lil", "csr", "ell", "dense"],
    "test_NonUniformDelay":                     ["lil", "csr"],
    "test_SynapseOperations":                   ["lil"],
    "test_SynapticAccess":                      ["lil", "csr"],
    # test_Monitor
    "test_MonitorRatePSP":                      ["lil"],
    "test_MonitorLocalVariable":                ["lil"],
    # SpecificProjections
    "test_Convolution":                         ["lil"],
    "test_Pooling":                             ["lil"],
    "test_CurrentInjection":                    ["lil"],
}

cuda = {
    # test_RateTransmission.py
    "test_RateTransmission":                ["csr", "ellr", "sell", "dense"],
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
    "test_NoDelay":                         ["csr", "ellr", "dense"],
    "test_UniformDelay":                    ["csr", "ellr", "dense"],
    "test_SynapseOperations":               ["csr", ],
    "test_SynapticAccess":                  ["csr", "ellr"],
    # test_Monitor
    "test_MonitorRatePSP":                  ["csr"],
    "test_MonitorLocalVariable":            ["csr"],
    # SpecificProjections
    "test_Convolution":                     ["csr"],
    "test_Pooling":                         ["csr"],
    "test_CurrentInjection":                ["csr"],
}
