from ANNarchy.core.Global import _check_paradigm

from .test_BuiltinFunctions import test_BuiltinFunctions
from .test_Connectivity import test_Connectivity
from .test_CustomFunc import test_CustomFunc
from .test_Dendrite import test_Dendrite
from .test_GlobalOperation import test_GlobalOps_1D, test_GlobalOps_2D, test_SynapticAccess
from .test_ITE import test_ITE
from .test_NeuronUpdate import test_LocalUpdate, test_GlobalUpdate, test_MixedUpdate
from .test_NumericalMethod import test_Explicit, test_Exponential, test_Implicit, test_Midpoint, test_ImplicitCoupled, test_MidpointCoupled, test_Precision
from .test_Population import test_Population1D, test_Population2D, test_Population3D, test_Population2x3D
from .test_PopulationView import test_PopulationView
from .test_Projection import test_Projection
from .test_RandomVariables import test_NeuronRandomVariables, test_SynapseRandomVariables
from .test_Record import test_Record
from .test_RateTransmission import test_RateTransmission
from .test_SpikingNeuron import test_SpikingCondition
from .test_SpikingSynapse import test_PreSpike, test_PostSpike

# Some features and accordingly Unittests are only allowed on specific platforms
if _check_paradigm('openmp'):
    from .test_StructuralPlasticity import test_StructuralPlasticityEnvironment, test_StructuralPlasticityModel
