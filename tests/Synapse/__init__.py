import unittest
from ANNarchy.core.Global import _check_paradigm, _check_precision, config

# Basic object accessors
from .test_RateTransmission import (test_RateTransmission,
                                    test_CustomConnectivityNoDelay, 
                                    test_CustomConnectivityUniformDelay)
from .test_Dendrite import test_DendriteDefaultSynapse, test_DendriteModifiedSynapse
from .test_Projection import test_Projection

# Operations
from .test_RateSynapse import test_Locality, test_AccessPSP, test_ModifiedPSP
from .test_RateDelays import (test_NoDelay, test_SynapseOperations,
                              test_SynapticAccess, test_UniformDelay)
from .test_SpikingSynapse import test_PreSpike, test_PostSpike
from .test_SpikingTransmission import test_SpikeTransmissionNoDelay, test_SpikeTransmissionUniformDelay
from .test_ContinuousUpdate import test_RateCodedContinuousUpdate, test_SpikingContinuousUpdate

# Other specific obects
from .test_SpecificProjections import test_CurrentInjection

# Some features and accordingly Unittests are only allowed on specific platforms
if _check_paradigm('openmp'):
    from .test_RateDelays import test_NonuniformDelay
    from .test_RateTransmission import test_CustomConnectivityNonUniformDelay
    from .test_SpikingTransmission import test_SpikeTransmissionNonUniformDelay
    from .test_StructuralPlasticity import test_StructuralPlasticityEnvironment, test_StructuralPlasticityModel
    from .test_Convolution import test_Convolution
    from .test_Pooling import test_Pooling

# Contains mapping which formats are allowed for which operation
from .storage_formats import single_thread, open_mp, cuda, p2p

def run_with(c, formats, orders):
    """
    Run the tests with all given storage formats and orders. This is achieved
    by copying the classes for every data format.
    """
    for s_format in formats:
        for s_order in orders:
            if s_order == "pre_to_post" and s_format not in ["lil", "csr"]:
                # pre_to_post does only work with formats lil and csr
                continue
            # append the class name with storage_format and storage_order
            cls_name = c.__name__ + "_" + str(s_format) + "_" + str(s_order)
            # Define a dict of globals that are used in the copied classes to
            # identify the currently tested format and storage_order
            glob = {"storage_format":s_format, "storage_order":s_order}
            # Add a new class to the python globals by creating a class with
            # name cls_name that inherits from the base class and the basic
            # unittest.TestCase class. It uses glob as its namespace.
            globals()[cls_name] = type(cls_name, (c, unittest.TestCase), glob)
    # Delete the base class so that it will not run without overloading.
    del globals()[c.__name__]
    del c

# Set the mode variable from one of three dictionaries for the current tests,
# as the data_formats are different with different paradigms.
if _check_paradigm('openmp'):
    if config['num_threads'] == 1:
        mode = single_thread
    else:
        mode = open_mp
else:
    mode = cuda
testCases = [t for t in locals().keys() if t in mode]

for case in testCases:
    # Default ordering in ANNarchy
    storage_orders = ["post_to_pre"]

    # In some cases we need to test the transposed view
    # This concerns data structurs intended for spiking-models.
    if case in p2p:
        storage_orders.append("pre_to_post")

    # Run the test case with the given modes and storage_orders
    run_with(globals()[case], mode[case], storage_orders)
