/*
 *    VecTransformation.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2023  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
#include <vector>

/**
 * @brief   Flatten a two-dimensional data structure.
 *
 * @tparam  VT
 * @param   data
 * @return  std::vector<VT>
 */
template<typename VT>
std::vector<VT> transform_2d_to_1d(std::vector<std::vector<VT>> data) {
    std::vector<VT> result;

    for (auto it = data.begin(); it < data.end();  it++) {
        result.insert(result.end(), it->begin(), it->end());
    }
    return result;
}

/**
 * @brief   Flatten a three-dimensional data structure.
 *
 * @tparam  VT
 * @param   data
 * @return  std::vector<VT>
 */
template<typename VT>
std::vector<VT> transform_3d_to_1d(std::vector<std::vector<std::vector<VT>>> data) {
#ifdef _DEBUG
    std::cout << "Flatting 3-dimensional structure (" << data.size() << "," << data[0].size() << "," << data[0][0].size() << ")" << std::endl;
#endif
    std::vector<VT> result;

    for (auto it = data.begin(); it < data.end();  it++) {
        for (auto it2 = it->begin(); it2 < it->end();  it2++) {
            result.insert(result.end(), it2->begin(), it2->end());
        }
    }
#ifdef _DEBUG
    std::cout << "returning 1-dimensional vector of size: " << result.size() << std::endl;
#endif
    result.shrink_to_fit(); // ensure that size == capacity
    return result;
}

/**
 * @brief   Flatten a four-dimensional data structure.
 *
 * @tparam  VT
 * @param   data
 * @return  std::vector<VT>
 */
template<typename VT>
std::vector<VT> transform_4d_to_1d(std::vector<std::vector<std::vector<std::vector<VT>>>> data) {
#ifdef _DEBUG
    std::cout << "Flatting 4-dimensional structure (" << data.size() << "," << data[0].size() << "," << data[0][0].size() << "," << data[0][0][0].size() << ")" << std::endl;
#endif
    std::vector<VT> result;

    for (auto it = data.begin(); it < data.end();  it++) {
        for (auto it2 = it->begin(); it2 < it->end();  it2++) {
            for (auto it3 = it2->begin(); it3 < it2->end();  it3++) {
                result.insert(result.end(), it3->begin(), it3->end());
            }
        }
    }
#ifdef _DEBUG
    std::cout << "returning 1-dimensional vector of size: " << result.size() << std::endl;
#endif
    result.shrink_to_fit(); // ensure that size == capacity
    return result;
}

/**
 * @brief   Flatten a five-dimensional data structure.
 *
 * @tparam  VT
 * @param   data
 * @return  std::vector<VT>
 */
template<typename VT>
std::vector<VT> transform_5d_to_1d(std::vector<std::vector<std::vector<std::vector<std::vector<VT>>>>> data) {
#ifdef _DEBUG
    std::cout << "Flatting 5-dimensional structure (" << data.size() << "," << data[0].size() << "," << data[0][0].size() << "," << data[0][0][0].size() << "," << data[0][0][0][0].size() << ")" << std::endl;
#endif
    std::vector<VT> result;

    for (auto it = data.begin(); it < data.end();  it++) {
        for (auto it2 = it->begin(); it2 < it->end();  it2++) {
            for (auto it3 = it2->begin(); it3 < it2->end();  it3++) {
                for (auto it4 = it3->begin(); it4 < it3->end();  it4++) {
                    result.insert(result.end(), it4->begin(), it4->end());
                }
            }
        }
    }
#ifdef _DEBUG
    std::cout << "returning 1-dimensional vector of size: " << result.size() << std::endl;
#endif
    result.shrink_to_fit(); // ensure that size == capacity
    return result;
}
