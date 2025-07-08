/*
 *    helper_functions.cuh
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2025  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
 */
#pragma once

#include "helper_functions.hpp"

/**
 *  @brief      check if the matrix fits into RAM
 *  @details    Unlike CUDA it appears that the standard C++ API does not
 *              provide a function to get the available RAM at a present time.
 *              Many sources recommended to use the /proc/meminfo file
 *  @todo       1) I'm not completely sure, what we want to do if /proc/meminfo is not readable. This should normally
 *                 not happen as far as I know. Currently, we would hope for the best ...
 *              2) The function is currently only implemented for Linux, while for Windows/MacOS we simply return true ...
 */
inline bool check_free_memory_cuda(size_t required) {

    size_t gpu_free, gpu_total;
    cudaMemGetInfo( &gpu_free, &gpu_total );
#ifdef _DEBUG
    std::cout << "Allocate " << required << " and have " << gpu_free << "( " << (double(required)/double(gpu_total)) * 100.0 << " percent of total memory)" << std::endl;
#endif

    // must have enough memory on CPU and GPU
    return check_free_memory(required) & (required < gpu_free) ;
}