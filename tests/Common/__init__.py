from ANNarchy.intern.ConfigManagement import _check_precision

from .test_ITE import test_ITE
from .test_NumericalMethod import (test_Explicit, test_Exponential,
                                   test_Implicit, test_Midpoint,
                                   test_ImplicitCoupled, test_MidpointCoupled,
                                   test_Precision)
from .test_BuiltinFunctions import test_BuiltinFunctions
from .test_CustomFunc import test_CustomFunc
from .test_SimulateUntil import test_SingleCondition, test_TwoConditionWithOr, test_TwoConditionWithAnd