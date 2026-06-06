"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
from dataclasses import dataclass

class CTypeBase:
    """
    Base class for CTypes used in our framework. All functions should be
    overloaded by the deriving classes, otherwise an exception is raised.
    """
    @property
    def cpp_decl_type(self) -> str:
        "Returns data type used int the C++ simulation code as string."
        raise NotImplementedError

    @property
    def py_decl_type(self) -> str:
        "Returns data type used in the nanobind interface to the C++ simulation code as string."
        raise NotImplementedError

    @property
    def bits(self) -> int:
        "Returns number of bits, raises exception if not implemented by child type."
        raise NotImplementedError

#
#   Floating-point precision type
#
@dataclass(frozen=True)
class FloatType(CTypeBase):
    """
    Floating-point data type.
    """
    exp: int
    mantissa: int

    FP32 = (8, 23)
    FP64 = (11, 52)

    @property
    def cpp_decl_type(self) -> str:
        "Returns data type used int the C++ simulation code as string."
        if (self.exp, self.mantissa) == FloatType.FP32:
            return "float"

        if (self.exp, self.mantissa) == FloatType.FP64:
            return "double"

        raise ValueError(
           f"Unsupported float format: exp={self.exp}, mantissa={self.mantissa}"
        )

    @property
    def py_decl_type(self) -> str:
        "Returns data type used in the nanobind interface to the C++ simulation code as string."
        if (self.exp, self.mantissa) == FloatType.FP32:
            # ANNarchy4.x behavior force to double always
            return "double"

        if (self.exp, self.mantissa) == FloatType.FP64:
            return "double"

        raise ValueError(
           f"Unsupported float format: exp={self.exp}, mantissa={self.mantissa}"
        )

    @property
    def bits(self) -> int:
        return 1 + self.exp + self.mantissa

# Some pre-defined types
float32 = FloatType(8, 23)
float64 = FloatType(11, 52)
