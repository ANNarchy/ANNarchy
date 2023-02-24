/*
 *    VecTransformation.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    Copyright (C) 2023  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
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
    std::vector<VT> result;

    for (auto it = data.begin(); it < data.end();  it++) {
        for (auto it2 = it->begin(); it2 < it->end();  it2++) {
            result.insert(result.end(), it2->begin(), it2->end());
        }
    }
    return result;
}