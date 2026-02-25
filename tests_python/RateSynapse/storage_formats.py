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
    "test_RateTransmission": ["lil", "auto", "csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_CustomConnectivityUniformDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_CustomConnectivityNonUniformDelay": ["lil", "auto", "csr", "ell"],
    # test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate": ["lil", "auto", "csr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels": ["lil", "auto", "csr", "dense"],
    # from test_Projection.py
    "test_DefaultProjection": ["lil", "auto", "csr", "dense", "bsr"],
    "test_ModifiedProjection": ["lil", "auto", "csr", "dense"],
    "test_SliceProjections": ["lil", "auto", "csr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse": ["lil", "auto", "csr", "ell", "dense"],
    "test_DendriteModifiedSynapse": ["lil", "auto", "csr"],
    # from test_RateSynapse.py
    "test_Locality": ["lil", "auto", "csr", "dense"],
    "test_AccessPSP": ["lil", "auto", "csr", "dense"],
    "test_ModifiedPSP": ["lil", "auto", "csr", "dense"],
    # from test_RateDelays
    "test_NoDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_UniformDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_NonUniformDelay": ["lil", "auto", "csr"],
    "test_SynapseOperations": ["lil", "auto"],
    "test_SynapticAccess": ["lil", "auto", "csr"],
    # test_Monitor
    "test_MonitorRatePSP": ["lil", "auto"],
    "test_MonitorLocalVariable": ["lil", "auto"],
    # SpecificProjections
    "test_Convolution": ["lil", "auto"],
    "test_Pooling": ["lil", "auto"],
    "test_CurrentInjection": ["lil", "auto"],
}

open_mp = {
    # test_RateTransmission.py
    "test_RateTransmission": ["lil", "auto", "csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_CustomConnectivityUniformDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_CustomConnectivityNonUniformDelay": ["lil", "auto", "csr", "ell"],
    # test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate": ["lil", "auto", "csr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels": ["lil", "auto", "csr", "dense"],
    # from test_Projection.py
    "test_DefaultProjection": ["lil", "auto", "csr", "dense", "bsr"],
    "test_ModifiedProjection": ["lil", "auto", "csr", "dense"],
    "test_SliceProjections": ["lil", "auto", "csr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse": ["lil", "auto", "csr", "ell", "dense"],
    "test_DendriteModifiedSynapse": ["lil", "auto", "csr"],
    # from test_RateSynapse.py
    "test_Locality": ["lil", "auto", "csr", "dense"],
    "test_AccessPSP": ["lil", "auto", "csr", "dense"],
    "test_ModifiedPSP": ["lil", "auto", "csr", "dense"],
    # from test_RateDelays
    "test_NoDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_UniformDelay": ["lil", "auto", "csr", "ell", "dense"],
    "test_NonUniformDelay": ["lil", "auto", "csr"],
    "test_SynapseOperations": ["lil", "auto"],
    "test_SynapticAccess": ["lil", "auto", "csr"],
    # test_Monitor
    "test_MonitorRatePSP": ["lil", "auto"],
    "test_MonitorLocalVariable": ["lil", "auto"],
    # SpecificProjections
    "test_Convolution": ["lil", "auto"],
    "test_Pooling": ["lil", "auto"],
    "test_CurrentInjection": ["lil", "auto"],
}

cuda = {
    # test_RateTransmission.py
    "test_RateTransmission": ["csr", "ellr", "sell", "dense"],
    # from test_RateCustomConnectivity.py
    "test_CustomConnectivityNoDelay": ["csr", "ellr"],
    "test_CustomConnectivityUniformDelay": ["csr", "ellr"],
    # from test_RateContinuousUpdate.py
    "test_RateCodedContinuousUpdate": ["csr", "ellr", "dense"],
    # test_RateDefaultSynapseModels.py
    "test_RateDefaultSynapseModels": ["csr", "ellr"],
    # from test_Projection.py
    "test_DefaultProjection": ["csr", "ellr"],
    "test_ModifiedProjection": ["csr", "ellr"],
    "test_SliceProjections": ["csr", "ellr"],
    # from test_Dendrite.py
    "test_DendriteDefaultSynapse": ["csr", "ellr"],
    "test_DendriteModifiedSynapse": ["csr"],
    # from test_RateSynapse.py
    "test_Locality": ["csr", "ellr", "dense"],
    "test_AccessPSP": ["csr", "ellr", "dense"],
    "test_ModifiedPSP": ["csr", "ellr", "dense"],
    # from test_RateDelays
    "test_NoDelay": ["csr", "ellr", "dense"],
    "test_UniformDelay": ["csr", "ellr", "dense"],
    "test_SynapseOperations": ["csr", "dense"],
    "test_SynapticAccess": ["csr", "ellr"],
    # test_Monitor
    "test_MonitorRatePSP": ["csr"],
    "test_MonitorLocalVariable": ["csr"],
    # SpecificProjections
    "test_Convolution": ["csr"],
    "test_Pooling": ["csr"],
    "test_CurrentInjection": ["csr"],
}
