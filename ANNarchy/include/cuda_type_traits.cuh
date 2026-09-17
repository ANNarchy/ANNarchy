/*
 *    cuda_type_traits.cuh
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2026  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

/*
 * Provides a minimal interface for handling numerical values.
 * In particular, the semantics of min() follow the definition
 * provided by std::numeric_limits.
 */

template <typename T>
struct cuda_type_traits
{
    __device__ __forceinline__
    static T zero()
    {
        return T{0};
    }
};

/*
 * 64-bit floating point values
 */
template <>
struct cuda_type_traits<double>
{
    __device__ __forceinline__
    static double max()
    {
        return DBL_MAX;
    }

    static double min()
    {
        return DBL_MIN;
    }

    __device__ __forceinline__
    static double lowest()
    {
        return -DBL_MAX;
    }

    __device__ __forceinline__
    static double zero()
    {
        return 0.0;
    }
};

/*
 * 32-bit floating point values
 */
template <>
struct cuda_type_traits<float>
{
    __device__ __forceinline__
    static float max()
    {
        return FLT_MAX;
    }

    __device__ __forceinline__
    static float min()
    {
        return FLT_MIN;
    }

    __device__ __forceinline__
    static float lowest()
    {
        return -FLT_MAX;
    }

    __device__ __forceinline__
    static float zero()
    {
        return 0.0f;
    }
};

#if __has_include(<cuda_fp16.h>)
/*
 * 16-bit floating point values
 */
template <>
struct cuda_type_traits<__half>
{
    __device__ __forceinline__
    static __half max()
    {
        return __float2half(65504.0f);
    }

    __device__ __forceinline__
    static __half min()
    {
        return __float2half(6.10352e-5f);
    }

    __device__ __forceinline__
    static __half lowest()
    {
        return __float2half(-65504.0f);
    }

    __device__ __forceinline__
    static __half zero()
    {
        return __float2half(0.0f);
    }
};
#endif

#if __has_include(<cuda_bf16.h>)
/*
 * 16-bit brain-float values
 */
template <>
struct cuda_type_traits<__nv_bfloat16>
{
    __device__ __forceinline__
    static __nv_bfloat16 max()
    {
        return __float2bfloat16(3.38953139e38f);
    }

    __device__ __forceinline__
    static __nv_bfloat16 min()
    {
        return __float2bfloat16(1.17549435e-38f);
    }

    __device__ __forceinline__
    static __nv_bfloat16 lowest()
    {
        return __float2bfloat16(-3.38953139e38f);
    }

    __device__ __forceinline__
    static __nv_bfloat16 zero()
    {
        return __float2bfloat16(0.0f);
    }
};
#endif