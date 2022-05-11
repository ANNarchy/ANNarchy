import unittest
from ANNarchy.core.Global import _check_paradigm, _check_precision, config

# Basic object accessors
from .test_RateTransmission import test_RateTransmission, test_CustomConnectivity
from .test_Dendrite import test_DendriteDefaultSynapse, test_DendriteModifiedSynapse
# from .test_Convolution import test_Convolution
# from .test_Pooling import test_Pooling
from .test_Projection import test_Projection

# Operations
from .test_RateSynapse import test_Locality, test_AccessPSP, test_ModifiedPSP
from .test_RateDelays import (test_NoDelay, test_UniformDelay, test_NonuniformDelay,
                              test_SynapseOperations, test_SynapticAccess)
from .test_SpikingSynapse import test_PreSpike, test_PostSpike
from .test_SpikingTransmission import test_SpikeTransmissionNoDelay, test_SpikeTransmissionUniformDelay, test_SpikeTransmissionNonUniformDelay
from .test_ContinuousUpdate import test_ContinuousUpdate

# Other specific obects
from .test_SpecificProjections import test_CurrentInjection

from .storage_formats import single_thread, open_mp, cuda, p2p

# Some features and accordingly Unittests are only allowed on specific platforms
if _check_paradigm('openmp'):
    from .test_RateDelays import test_NonuniformDelay
    from .test_StructuralPlasticity import test_StructuralPlasticityEnvironment, test_StructuralPlasticityModel

def run_with(c, formats, orders):
    """
    Run the tests with all given storage formats and orders. This is achieved
    by copying the classes for every data format.
    """
    for s_format in formats:
        for s_order in orders:
            if s_order == "pre_to_post" and s_format not in ["lil", "csr"]:
                continue
            cls_name = c.__name__ + "_" + str(s_format) + "_" + str(s_order)
            glob = {"storage_format":s_format, "storage_order":s_order}
            globals()[cls_name] = type(cls_name, (c, unittest.TestCase), glob)
    # Delete the base class so that it will not be done again
    del globals()[c.__name__]
    del c

if _check_paradigm('openmp'):
    if config['num_threads'] == 1:
        mode = single_thread
    else:
        mode = open_mp
else:
    mode = cuda
testCases = [t for t in locals().keys() if t in mode]

for case in testCases:
    if case in p2p:
        storage_orders = ["pre_to_post", "post_to_pre"]
    else:
        storage_orders = ["post_to_pre",]
    run_with(globals()[case], mode[case], storage_orders)
