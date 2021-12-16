import unittest
from ANNarchy.core.Global import _check_paradigm, _check_precision

# Basic object accessors
from .test_Connectivity import TestConnectivity, TestCustomConnectivity
# from .test_Convolution import test_Convolution
from .test_Dendrite import test_Dendrite
# from .test_Pooling import test_Pooling
from .test_Projection import test_ProjectionLIL, test_ProjectionCSR, test_ProjectionCSR2

# Operations
from .test_RateSynapse import test_Locality, test_AccessPSP, test_ModifiedPSP
from .test_RateTransmission import test_NoDelay, test_UniformDelay, test_SynapseOperations, test_SynapticAccess
from .test_SpikingSynapse import test_PreSpike, test_PostSpike
from .test_SpikingTransmission import test_LILConnectivity
from .test_SynapseUpdate import test_SynapseUpdate

# Other specific obects
from .test_SpecificProjections import test_CurrentInjection

# Some features and accordingly Unittests are only allowed on specific platforms
if _check_paradigm('openmp'):
    from .test_RateTransmission import test_NonuniformDelay
    from .test_StructuralPlasticity import test_StructuralPlasticityEnvironment, test_StructuralPlasticityModel

def run_with(c, formats, orders):
    """
    Run the tests with all given storage formats and orders. This is achieved
    by copying the classes for every data format.
    """
    for s_format in formats:
        for s_order in orders:
            cls_name = c.__name__ + "_" + str(s_format) + "_" + str(s_order)
            glob = {"storage_format":s_format, "storage_order":s_order}
            globals()[cls_name] = type(cls_name, (c, unittest.TestCase), glob)
    # Delete the base class so that it will not be done again
    del globals()[c.__name__]
    del c

storage_formats = ["lil", "csr", "ell"]
storage_orders = ["pre_to_post", "post_to_pre"]

classes = [test_NoDelay, test_SynapticAccess, test_AccessPSP, test_ModifiedPSP,
           TestConnectivity]
for cl in classes:
    run_with(cl, ['lil', 'csr', 'ell'], ['pre_to_post', 'post_to_pre'])

run_with(test_Locality, ['lil', 'csr'], ['pre_to_post', 'post_to_pre'])

lil_classes = [test_Dendrite, test_UniformDelay, test_SynapseOperations,
               test_PreSpike, test_PostSpike, test_NonuniformDelay]
for cl in lil_classes:
    run_with(cl, ['lil'], ['pre_to_post', 'post_to_pre'])

# Just so to make sure, that the tests will not be run twice.
del classes
del lil_classes
del storage_formats
del storage_orders
