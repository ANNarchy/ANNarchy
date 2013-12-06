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
#ifndef __GLOBAL_H__
#define __GLOBAL_H__

//
//	stl and other common things
#include <string>
#include <vector>
#include <iostream>

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

#ifdef ANNAR_DOUBLE
    #define DATA_TYPE double
#else
    #define DATA_TYPE float
#endif

//
//	conventions
namespace ANNarchy_Global
{
    inline DATA_TYPE positive(DATA_TYPE x) { return x > 0.0 ? x : 0.0; }

    inline DATA_TYPE negative(DATA_TYPE x) { return x < 0.0 ? x : 0.0; }

    static int time;
}
//#define ANNAR_PROFILE

#include "Random.h"
#include "Network.h"
#include "Population.h"
#include "Connector.h"
#include "Projection.h"

#endif
