"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

header_template = """#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <random>
#include <cassert>
// only included if compiled with -fopenmp
#ifdef _OPENMP
    #include <omp.h>
#endif

// Intrinsic operations (Intel/AMD)
#ifdef __x86_64__
    #include <immintrin.h>
#endif

// Useful functions
#include "logging.hpp"
#include "helper_functions.hpp"

/*
 * Built-in functions
 *
 */
%(built_in)s

/*
 * Custom constants
 *
 */
%(custom_constant)s

/*
 * Custom functions
 *
 */
%(custom_func)s

/*
 * Structures for the populations
 *
 */
%(pop_struct)s
/*
 * Structures for the projections
 *
 */
%(proj_struct)s

/*
 * Declaration of the populations
 *
 */
%(pop_ptr)s

/*
 * Declaration of the projections
 *
 */
%(proj_ptr)s

/*
 * Recorders
 *
 */
#include "Monitor.hpp"

extern std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(Monitor* recorder);
size_t estimate_record_size(int num_steps);

/*
 * Simulation methods
 *
 */
void run(const int nbSteps);
int run_until(const int steps, std::vector<int> populations, bool or_and);
void step();

/*
 *  Initialization
 */
void initialize(const %(float_prec)s dt_) ;

/*
 *  Life-time management
 */
void create_cpp_instances();
void destroy_cpp_instances();

/*
 * Time export
 *
 */
long int getTime();
void setTime(const long int t_);
%(float_prec)s getDt();
void setDt(const %(float_prec)s dt_);

/*
 * Number of threads
 *
 */
void setNumberThreads(int threads, std::vector<int> core_list);

/*
 * Seed for the RNG
 *
*/
void setSeed(long int seed, int num_sources, bool use_seed_seq);
"""

body_template = """
#include "ANNarchy.hpp"

%(prof_include)s

/*
 * Internal data
 *
 */
%(float_prec)s dt;
long int t;
std::vector<std::mt19937> rng;

// Custom constants
%(custom_constant)s

// Populations
%(pop_ptr)s

// Projections
%(proj_ptr)s

// Global operations
%(glops_def)s

/*
 * Recorders
 */
std::vector<Monitor*> recorders;
int addRecorder(Monitor* recorder){
    int found = -1;

    for (unsigned int i=0; i<recorders.size(); i++) {
        if (recorders[i] == nullptr) {
            found = i;
            break;
        }
    }

    if (found != -1) {
        // fill a previously cleared slot
        recorders[found] = recorder;
        return found;
    } else {
        recorders.push_back(recorder);
        return recorders.size() - 1;
    }
}

Monitor* getRecorder(int id) {
    if (id < recorders.size())
        return recorders[id];
    else
        return nullptr;
}

void removeRecorder(Monitor* recorder){
    for (unsigned int i=0; i<recorders.size(); i++){
        if (recorders[i] == recorder) {
            // mark the slot as free
            recorders[i] = nullptr;
            break;
        }
    }
}

size_t estimate_record_size(int num_steps) {
    size_t estimate = 0;
    for (unsigned int i=0; i < recorders.size(); i++){
        if (recorders[i])
            estimate += recorders[i]->estimate_size_in_bytes(num_steps);
    }
    return (estimate / 1024 / 1024);    // return in MiB for easier handling
}

/*
 *  Simulation methods
 */
// Simulate a single step
void singleStep()
{
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- evaluate step " << t << " (" << t * dt << " ms) --" << std::endl;
#endif
%(prof_step_pre)s

    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////
%(prof_proj_psp_pre)s
%(reset_sums)s
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update psp/conductances ..." << std::endl;
#endif
%(compute_sums)s
%(prof_proj_psp_post)s

    ////////////////////////////////
    // Recording target variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Record psp/conductances ..." << std::endl;
#endif
    for (unsigned int i=0; i < recorders.size(); i++) {
        if (recorders[i])
            recorders[i]->record_targets();
    }

    ////////////////////////////////
    // Update random distributions
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Draw required random numbers ..." << std::endl;
#endif
%(prof_rng_pre)s
%(random_dist_update)s
%(prof_rng_post)s

    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate neural ODEs ..." << std::endl;
#endif
%(prof_neur_step_pre)s
%(update_neuron)s
%(prof_neur_step_post)s

    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update delay queues ..." << std::endl;
#endif
%(delay_code)s

    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Update global operations ..." << std::endl;
#endif
%(prof_global_ops_pre)s
%(update_globalops)s
%(prof_global_ops_post)s

    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Evaluate synaptic ODEs ..." << std::endl;
#endif
%(prof_proj_step_pre)s
%(update_synapse)s
%(prof_proj_step_post)s

    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////
%(prof_proj_post_event_pre)s
%(post_event)s
%(prof_proj_post_event_post)s

    ////////////////////////////////
    // Structural plasticity
    ////////////////////////////////
%(structural_plasticity)s

    ////////////////////////////////
    // Recording neural / synaptic variables
    ////////////////////////////////
%(prof_record_pre)s
    for (unsigned int i=0; i < recorders.size(); i++){
        if (recorders[i])
            recorders[i]->record();
    }
%(prof_record_post)s

    ////////////////////////////////
    // Increase internal time
    ////////////////////////////////
    t++;

%(prof_step_post)s

#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- simulation step " << t << " completed --" << std::endl;
#endif
}

// Simulate the network for the given number of steps,
// called from python
void run(const int nbSteps) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "Perform simulation for " << nbSteps << " steps." << std::endl;
#endif

%(prof_run_pre)s
    // apply changes implied by structural plasticity (spike only)
%(sp_spike_backward_view_update)s

    // perform the simulation
    for(int i=0; i<nbSteps; i++) {
        singleStep();
    }
%(prof_run_post)s
}

// Simulate the network for a single steps,
// called from python
void step() {
%(prof_run_pre)s
    // apply changes implied by structural plasticity (spike only)
%(sp_spike_backward_view_update)s

    // perform a single step (size dt)
    singleStep();
%(prof_run_post)s
}

int run_until(const int steps, std::vector<int> populations, bool or_and)
{
    // apply changes implied by structural plasticity (spike only)
%(sp_spike_backward_view_update)s

    // perform the simulation until the condition is satisfied
%(run_until)s
}

/*
 *  Initialization methods
 */
// Initialize the internal data and the random numbers generator
void initialize(const %(float_prec)s _dt) {
%(initialize)s
}

// Change the seed of the RNG
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq) {
    std::string msg = "ANNarchyCore::setSeed(): " + std::to_string(seed) + ", " + std::to_string(num_sources) + ", " + std::string((use_seed_seq) ? "true" : "false");
    ANNARCHY_LOG_MSG(msg);

    // sanity check
    if (num_sources > 1)
        std::cerr << "WARNING - ANNarchyCore::setSeed(): num_sources should be 1 for single thread code." << std::endl;

    rng.clear();

    rng.push_back(std::mt19937(seed));

    rng.shrink_to_fit();
}

/*
 *  Life-time management
 */
void create_cpp_instances() {
#if defined(_TRACE_INIT) || defined(_DEBUG)
    std::cout << "Instantiate C++ objects ..." << std::endl;
#endif
}

void destroy_cpp_instances() {
#if defined(_TRACE_INIT) || defined(_DEBUG)
    std::cout << "Destroy C++ objects ..." << std::endl;
#endif
}

/*
 * Access to time and dt
 */
long int getTime() {return t;}
void setTime(const long int t_) { t=t_;}
%(float_prec)s getDt() { return dt;}
void setDt(const %(float_prec)s dt_) { dt=dt_;}

/*
 * Number of threads
 *
*/
void setNumberThreads(const int threads, const std::vector<int> core_list)
{
    if (threads > 1) {
        std::cerr << "WARNING: a call of setNumberThreads() is without effect on single thread simulation code." << std::endl;
    }

    if (core_list.size()>1) {
        std::cerr << "The provided core list is ambiguous and therefore ignored." << std::endl;
        return;
    }

#ifdef __linux__
    // set a cpu mask to prevent moving of threads
    cpu_set_t mask;

    // no CPUs selected
    CPU_ZERO(&mask);

    // no proc_bind
    for(auto it = core_list.begin(); it != core_list.end(); it++)
        CPU_SET(*it, &mask);
    const int set_result = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
#else
    if (!core_list.empty()) {
        std::cout << "WARNING: manipulation of CPU masks is only available for linux systems." << std::endl;
    }
#endif
}
"""

run_until_template = {
    "default": """
    run(steps);
    return steps;
""",
    "body": """
    bool stop = false;
    bool cond_activated = false;
    int nb = 0;
    for(int n = 0; n < steps; n++)
    {
        step();
        nb++;
        stop = or_and;

%(run_until)s

        // HD: stop will be automatically true, if no populations are checked.
        if(stop && (populations.size() > 0))
            break;
    }
    return nb;

""",
    "single_pop": """
        cond_activated = std::find(populations.begin(), populations.end(), %(id)s) != populations.end();
        if (cond_activated)
            if(or_and)
                stop = stop && pop%(id)s->stop_condition();
            else
                stop = stop || pop%(id)s->stop_condition();
    """,
}

initialize_template = """
%(prof_init)s
    // Internal variables
    dt = _dt;
    t = static_cast<long int>(0);

    // Populations
%(pop_init)s

    // Projections
%(proj_init)s

    // Custom constants
%(custom_constant)s
"""

built_in_functions = """
#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) % long(b)
#define ite(a, b, c) (a?b:c)
"""

integer_power = """
// power function for integer exponent
inline %(float_prec)s power(%(float_prec)s x, unsigned int a){
    %(float_prec)s res=x;
    for (unsigned int i=0; i< a-1; i++){
        res *= x;
    }
    return res;
};
"""
