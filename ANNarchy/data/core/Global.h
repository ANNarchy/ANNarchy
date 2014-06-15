/*
 *    Global.h
 *
 *    This file is part of ANNarchy.
 *   
 *   Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
 *   Helge Ülo Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   ANNarchy is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __ANNARCHY_GLOBAL_H__
#define __ANNARCHY_GLOBAL_H__

//
//	stl and other common things
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <deque>
#include <iostream>
#include <sstream>
#include <fstream>

//
//	platform dependencies
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <stdio.h>
#endif

//
//	parallel stuff
#include <omp.h>

//
//	conventions
namespace ANNarchy_Global
{
    inline DATA_TYPE positive(DATA_TYPE x) { return x > 0.0 ? x : 0.0; }

    inline DATA_TYPE negative(DATA_TYPE x) { return x < 0.0 ? x : 0.0; }

    inline DATA_TYPE clip(DATA_TYPE x, DATA_TYPE a, DATA_TYPE b) { return (x < a ? a : (x > b ? b : x)); }

    extern int time;

    /**
     * 	\brief		simple function to split a string at a delimiter
     * 	\details	I'm still wondering, when something trivial like that, will be part of STL ...
     * 	\param[in]	string to split
     * 	\param[in]	char working as delimiter
     */
    inline std::vector<std::string> split(const std::string s, char delim)
    {
    	std::vector<std::string> elems;

        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }

        return elems;
    }

    //FUNCTIONS
}

//
// additional defines for more deeply debugging of several parts of ANNarchy framework
// #define _DEBUG_DELAY
// #define _DEBUG_DENDRITE_DATA
// #define _DEBUG_SIMULATION_CONTROL	// prove correctness of function calls, etc.
// #define _DEBUG_PARALLELISM			// prove correctness of parallelization / partition division

//
// profiling extension
// #define ANNAR_PROFILE
// #define ANNAR_SCHEDULE
// #define _DEBUG_PROFILER				// prove functionality of profiling extension
#ifdef ANNAR_PROFILE
#include "Profile.h"
#endif

#include "Random.h"
#include "Network.h"
#include "Population.h"
#include "Projection.h"
#include "Dendrite.h"

#endif
