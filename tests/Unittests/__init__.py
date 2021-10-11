from ANNarchy.core.Global import _check_paradigm, _check_precision

# Equations
from .test_ITE import test_ITE
from .test_NumericalMethod import test_Explicit, test_Exponential, test_Implicit, test_Midpoint, test_ImplicitCoupled, test_MidpointCoupled, test_Precision
from .test_BuiltinFunctions import test_BuiltinFunctions

# Basic object accessors
from .test_connectivity import TestConnectivity, TestCustomConnectivity
from .test_CustomFunc import test_CustomFunc
from .test_Dendrite import test_Dendrite
from .test_Population import test_Population1D, test_Population2D, test_Population3D, test_Population2x3D
from .test_PopulationView import test_PopulationView
from .test_Projection import test_ProjectionLIL, test_ProjectionCSR, test_ProjectionCSR2

# Operations
from .test_NeuronUpdate import test_NeuronUpdate
from .test_GlobalOperations import test_GlobalOps_1D, test_GlobalOps_1D_Large, test_GlobalOps_2D
from .test_RateTransmission import test_LIL_NoDelay, test_LIL_UniformDelay, test_LIL_GlobalOperations, test_SynapticAccess
from .test_SpikingTransmission import test_LILConnectivity
from .test_SpikingNeuron import test_SpikingCondition
from .test_Synapse import test_Locality, test_AccessPSP
from .test_SpikingSynapse import test_PreSpike, test_PostSpike

# Other specific obects
from .test_Record import test_Record
from .test_TimedArray import test_TimedArray
from .test_SpecificProjections import test_CurrentInjection

if _check_precision('double'):
    from .test_RandomVariables import test_NeuronRandomVariables, test_SynapseRandomVariables

# Some features and accordingly Unittests are only allowed on specific platforms
if _check_paradigm('openmp'):
    from .test_RateTransmission import test_LIL_NonuniformDelay
    from .test_StructuralPlasticity import test_StructuralPlasticityEnvironment, test_StructuralPlasticityModel
