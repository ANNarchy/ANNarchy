/*
 *    helper_functions.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
 *                        Julien Vitay <julien.vitay@gmail.com>
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

/**
 * @brief           Sort two vectors of same length.
 * @details         Sort criterion must be in a. The values are sorted ascending.
 * @tparam Type1    Data type for the first vector. Must be iterable and support a sorting.
 * @tparam Type2    Data type for the second vector.
 * @param a         first vector defines the sort base.
 * @param b         second vector will be sorted according to a.
 * @param n         number of elements (must be the same for both vectors).
 */
template<typename Type1, typename Type2>
void pairsort(Type1 *a, Type2 *b, int n)
{
    std::pair<Type1, Type2> pairt[n];

    // Store all elements in pairs.
    for (int i = 0; i < n; i++)
    {
        pairt[i].first = a[i];
        pairt[i].second = b[i];
    }

    // Sorting the pair array.
    std::sort(pairt, pairt + n);

    // Modifying original arrays
    for (int i = 0; i < n; i++)
    {
        a[i] = pairt[i].first;
        b[i] = pairt[i].second;
    }
}

/**
 *  @brief      check if the matrix fits into RAM
 *  @details    Unlike CUDA it appears that the standard C++ API does not
 *              provide a function to get the available RAM at a present time.
 *              Many sources recommended to use the /proc/meminfo file
 *  @todo       1) I'm not completely sure, what we want to do if /proc/meminfo is not readable. This should normally
 *                 not happen as far as I know. Currently, we would hope for the best ...
 *              2) The function is currently only implemented for Linux, while for Windows/MacOS we simply return true ...
 */
inline bool check_free_memory(size_t required) {
#ifdef __linux__
    FILE *meminfo = fopen("/proc/meminfo", "r");

    // TODO 1
    if(meminfo == nullptr) {
        std::cerr << "Could not read '/proc/meminfo'. ANNarchy can not catch to large allocations ..." << std::endl;
        return true;
    }

    char line[256];
    int ram;

    while(fgets(line, sizeof(line), meminfo))
    {
        if(sscanf(line, "MemFree: %d kB", &ram) == 1)
            break;  // hit
    }

    fclose(meminfo);
    size_t available = static_cast<size_t>(ram) * 1024;
#ifdef _DEBUG
    std::cout << "  we will allocate on CPU " << required << " from " << available << " bytes " << std::endl;
#endif
    
    return required < available;        
#else
    // TODO 2
    return true;
#endif
}