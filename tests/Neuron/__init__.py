# Basic object accessors
from .test_Population import (test_Population1D, test_Population2D,
                              test_Population3D, test_Population2x3D)
from .test_PopulationView import test_PopulationView
from .test_IndividualNeuron import test_IndividualNeuron

# Operations
from .test_GlobalOperations import (test_GlobalOps_1D, test_GlobalOps_1D_Large,
                                    test_GlobalOps_2D, test_GlobalOps_MultiUse)
from .test_NeuronUpdate import test_NeuronUpdate
from .test_SpikingNeuron import test_SpikingCondition

# Recording
from .test_Monitor import test_PopulationMonitor