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

//
//	conventions
#ifdef ANNAR_DOUBLE
	#define DATA_TYPE double
	#define positive(x) x > 0.0 ? x : 0.0
#else
	#define DATA_TYPE float
	#define positive(x) x > 0.0f ? x : 0.0f
#endif

#ifdef ANNAR_DOUBLE
	#define DATA_TYPE double
	#define negative(x) x < 0.0 ? x : 0.0
#else
	#define DATA_TYPE float
	#define negative(x) x < 0.0f ? x : 0.0f
#endif

//#define ANNAR_PROFILE

#include "Random.h"
#include "Network.h"
#include "Population.h"
#include "Connector.h"
#include "Projection.h"

#endif
