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
    def py_decl_type(self) -> str:
        """
        Returns data type used in the nanobind interface to the C++ simulation code as string.
        """
        # Since the original ANNarchy 4.4.0 release we have the principle that the interface
        # solely uses double precision.
        #
        # TODO: In particular for low-precision types and FixedType it might be worth to check
        #       the number of fraction bits to decide between fp32 and fp64? Even though I believe
        #       that it will turn out as fp32 in most cases. I assume that the usage of a fixed-
        #       point type will be play only a role in small precision use cases
        #       (HD: June 5, 2026)
        return "double"

    @property
    def cpp_decl_type(self) -> str:
        "Returns data type used int the C++ simulation code as string."
        raise NotImplementedError

    @property
    def bits(self) -> int:
        "Returns number of bits, raises exception if not implemented by child type."
        raise NotImplementedError

    def __str__(self):
        "Returns a short descriptor string"
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

    # Pre-defined type combinations
    BF16 = (8, 7)
    FP16 = (5, 10)
    FP32 = (8, 23)
    FP64 = (11, 52)

    @property
    def cpp_decl_type(self) -> str:
        "Returns data type used int the C++ simulation code as string used for openMP codes."
        if (self.exp, self.mantissa) == FloatType.BF16:
            return "__nv_bfloat16"

        if (self.exp, self.mantissa) == FloatType.FP16:
            return "__half"

        if (self.exp, self.mantissa) == FloatType.FP32:
            return "float"

        if (self.exp, self.mantissa) == FloatType.FP64:
            return "double"

        raise ValueError(
           f"Unsupported float format: exp={self.exp}, mantissa={self.mantissa}"
        )

    def __str__(self):
        "Returns a short descriptor string"
        if (self.exp, self.mantissa) == FloatType.BF16:
            return "bfloat16"

        if (self.exp, self.mantissa) == FloatType.FP16:
            return "float16"

        if (self.exp, self.mantissa) == FloatType.FP32:
            return "float32"

        if (self.exp, self.mantissa) == FloatType.FP64:
            return "float64"

        raise ValueError(
           f"Unsupported float format: exp={self.exp}, mantissa={self.mantissa}"
        )

    @property
    def bits(self) -> int:
        return 1 + self.exp + self.mantissa

# Some pre-defined types
bfloat16 = FloatType(8, 7)  # only supported by GPUs yet
float16 = FloatType(5, 10)  # only supported by GPUs yet
float32 = FloatType(8, 23)
float64 = FloatType(11, 52)

__all__ = ["bfloat16", "float16", "float32", "float64"]
