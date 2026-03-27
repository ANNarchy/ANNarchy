/*
 *    logging.hpp
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

// Only errors are print-out (default)
#define LOG_LEVEL_NO_LOG    0
// Log object allocation/deallocation, update of variables
#define LOG_LEVEL_DEBUG     1
// Log almost all function calls, this includes in particular the CPP template library.
#define LOG_LEVEL_TRACE     2

// dependent on the set compiler flags, we set the debug level
#if defined(_TRACE_INIT) || defined (_TRACE_SIMULATION_STEPS)
    #define CURRENT_LOG_LEVEL LOG_LEVEL_TRACE
#elif defined(_DEBUG)
    #define CURRENT_LOG_LEVEL LOG_LEVEL_DEBUG
#else
    #define CURRENT_LOG_LEVEL LOG_LEVEL_NO_LOG
#endif

// This functions are intended only in the CPP template library
#if CURRENT_LOG_LEVEL == LOG_LEVEL_TRACE
    #define ANNARCHY_LOG_ALLOC(class_name, ptr) std::cout << "[ALLOC]   " << class_name << "::" << class_name << "(this=" << ptr << ")" << std::endl
    #define ANNARCHY_LOG_DEALLOC(class_name, ptr) std::cout << "[DEALLOC] " << class_name << "::~" << class_name << "(this=" << ptr << ")" << std::endl
    #define ANNARCHY_LOG_CALL(class_name, func_name, ptr) std::cout << "[CALL]    " << class_name << "::" << func_name << "(this=" << ptr << ")" << std::endl
    #define ANNARCHY_LOG_ARG(arg, printout) std::cout << "[ARGS]    - " << arg << ": " << printout << std::endl
    #define ANNARCHY_LOG_STATE(attr, printout) std::cout << "[STATE]   - " << attr << ": " << printout << std::endl
    #define ANNARCHY_LOG_MSG(msg) std::cout << "[DEBUG]   " << msg << std::endl
#elif CURRENT_LOG_LEVEL == LOG_LEVEL_DEBUG
    #define ANNARCHY_LOG_ALLOC(class_name, ptr) std::cout << "[ALLOC]   " << class_name << " has been allocated at position " << ptr << std::endl
    #define ANNARCHY_LOG_DEALLOC(class_name, ptr) std::cout << "[DEALLOC] " << class_name << " has been deallocated from position " << ptr << std::endl
    #define ANNARCHY_LOG_CALL(class_name, func_name, ptr) void (0)
    #define ANNARCHY_LOG_ARG(arg, printout) void (0)
    #define ANNARCHY_LOG_STATE(attr, printout) void (0)
    #define ANNARCHY_LOG_MSG(msg) std::cout << "[DEBUG]   " << msg << std::endl
#else
    // similar to assert()
    #define ANNARCHY_LOG_ALLOC(class_name, ptr) void (0)
    #define ANNARCHY_LOG_DEALLOC(class_name, ptr) void (0)
    #define ANNARCHY_LOG_CALL(class_name, func_name, ptr) void (0)
    #define ANNARCHY_LOG_ARG(arg, printout) void (0)
    #define ANNARCHY_LOG_STATE(attr, printout) void (0)
    #define ANNARCHY_LOG_MSG(msg) void (0)
#endif


// Tracking of simulation steps, etc. are only shown if debug and trace flags are set.
#if defined (_TRACE_SIMULATION_STEPS) && (CURRENT_LOG_LEVEL >= LOG_LEVEL_DEBUG)
    #define ANNARCHY_TRACE_SIM_MSG(msg) std::cout << "[SIM]     " << msg << std::endl
    // For OpenMP, we need to distinguish between master- and worker-threads.
    #define ANNARCHY_TRACE_SIM_MSG_MASTER(msg) _Pragma("omp single") { \
        std::cout << "[SIM]     " << msg << std::endl; \
        std::cout << std::flush; \
    }

    #define ANNARCHY_TRACE_SIM_STEP(class_name, func_name, ptr) std::cout << "[SIM]     - " << class_name << "::" << func_name << "(this=" << ptr << ")" << std::endl
    // For OpenMP, we need to distinguish between master- and worker-threads.
    #define ANNARCHY_TRACE_SIM_STEP_MASTER(class_name, func_name, ptr) _Pragma("omp single") { \
        std::cout << "[SIM]     - " << class_name << "::" << func_name << "(this=" << ptr << ")" << std::endl; \
        std::cout << std::flush; \
    }
    #define ANNARCHY_TRACE_SIM_STEP_WORKER(class_name, func_name, ptr, tid, nt) _Pragma("omp critical") { \
        std::cout << "[SIM]     - " << class_name << "::" << func_name << "(this=" << ptr << "): tid=" << tid << ", nt=" << nt << std::endl; \
        std::cout << std::flush; \
    }

#else
    #define ANNARCHY_TRACE_SIM_MSG(msg) void(0)
    #define ANNARCHY_TRACE_SIM_MSG_MASTER(msg) void(0)
    #define ANNARCHY_TRACE_SIM_STEP(class_name, func_name, ptr) void (0)
    #define ANNARCHY_TRACE_SIM_STEP_MASTER(class_name, func_name, ptr) void (0)
    #define ANNARCHY_TRACE_SIM_STEP_WORKER(class_name, func_name, ptr, tid, nt) void (0)
#endif
