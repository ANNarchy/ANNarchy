"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

omp_header_template = """#pragma once

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

st_body_template = """
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
            // delete the present instance
            delete recorders[i];
            // mark the slot as free
            recorders[i] = nullptr;
            break;
        }
    }
}

/*
 *  Simulation methods
 */
// Simulate a single step
void singleStep()
{
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
    if (num_sources > 1)
        std::cerr << "WARNING - ANNarchy::setSeed(): num_sources should be 1 for single thread code." << std::endl;

    rng.clear();

    rng.push_back(std::mt19937(seed));

    rng.shrink_to_fit();
}

/*
 *  Life-time management
 */
void create_cpp_instances() {
#ifdef _DEBUG
    std::cout << "Instantiate C++ objects ..." << std::endl;
#endif
}

void destroy_cpp_instances() {
#ifdef _DEBUG
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

omp_body_template = """
#include "ANNarchy.hpp"

#ifdef __linux__
#include <sched.h>
#endif

%(prof_include)s

/*
 * Internal data
 *
 */
%(float_prec)s dt;
long int t;
std::vector<std::mt19937> rng;

// number openMP threads
int global_num_threads = -1;

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
            delete recorders[i];
            recorders[i] = nullptr;
            break;
        }
    }
}

/*
 *  Simulation methods
 */
// Step method. Generated by ANNarchy.
void singleStep(const int tid, const int nt)
{
%(prof_step_pre)s

    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////
%(prof_proj_psp_pre)s
%(reset_sums)s
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Update psp/conductances ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(compute_sums)s

    #pragma omp barrier
%(prof_proj_psp_post)s

    ////////////////////////////////
    // Recording target variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Record psp/conductances ..." << std::endl;
        std::cout << std::flush;
    }
#endif
    for (unsigned int i=tid; i < recorders.size(); i += nt) {
        if (recorders[i])
            recorders[i]->record_targets();
    }

    #pragma omp barrier

    ////////////////////////////////
    // Update random distributions
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Draw required random numbers ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(prof_rng_pre)s
%(random_dist_update)s
%(prof_rng_post)s

    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Evaluate neural ODEs ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(prof_neur_step_pre)s
%(update_neuron)s
%(prof_neur_step_post)s

    #pragma omp barrier

    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Update delay queues ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(delay_code)s

    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Update global operations ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(prof_global_ops_pre)s
%(update_globalops)s
%(prof_global_ops_post)s

    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Evaluate synaptic ODEs ..." << std::endl;
        std::cout << std::flush;
    }
#endif
%(prof_proj_step_pre)s
%(update_synapse)s
%(prof_proj_step_post)s

    #pragma omp barrier

    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////
%(post_event)s

    ////////////////////////////////
    // Structural plasticity
    ////////////////////////////////
%(structural_plasticity)s

    ////////////////////////////////
    // Recording neural / synaptic variables
    ////////////////////////////////
    #pragma omp barrier
#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "Recording state variables ..." << std::endl;
        std::cout << std::flush;
    }
#endif

%(prof_record_pre)s
    for (unsigned int i=tid; i < recorders.size(); i += nt){
        if (recorders[i])
            recorders[i]->record();
    }
%(prof_record_post)s

    #pragma omp barrier

    ////////////////////////////////
    // Increase internal time
    ////////////////////////////////
    #pragma omp single
    {
        t++;
    } // implicit barrier

%(prof_step_post)s

#ifdef _TRACE_SIMULATION_STEPS
    #pragma omp single
    {
        std::cout << "--- simulation step " << t << " completed ---" << std::endl;
        std::cout << std::flush;
    }
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
    #pragma omp parallel num_threads(global_num_threads)
    {
        int tid = omp_get_thread_num();

        for (int i=0; i<nbSteps; i++) {
            singleStep(tid, global_num_threads);
        }
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
    #pragma omp parallel num_threads(global_num_threads)
    {
        int tid = omp_get_thread_num();

        singleStep(tid, global_num_threads);
    }
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
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq){
#ifdef _DEBUG
    std::cout << "setSeed(): " << seed << ", " << num_sources << ", " << std::string((use_seed_seq) ? "true" : "false") << std::endl;
#endif
    rng.clear();

    if (num_sources == 1) {
        rng.push_back(std::mt19937(seed));
    } else {
        if (use_seed_seq) {
            std::seed_seq seq{seed};
            std::vector<std::uint32_t> seeds(num_sources);
            seq.generate(seeds.begin(), seeds.end());

            for (auto it = seeds.begin(); it != seeds.end(); it++) {
                rng.push_back(std::mt19937(*it));
            }
        } else {
            // Using seed initialization of M.E. O'Neill (randutils)
            std::vector<std::uint32_t> seeds(num_sources);
            randutils::auto_seed_256 seeder;
            seeder.generate(seeds.begin(), seeds.end());

            for (auto it = seeds.begin(); it != seeds.end(); it++) {
                rng.push_back(std::mt19937(*it));
            }
        }
    }

    rng.shrink_to_fit();
}

/*
 *  Life-time management
 */
void create_cpp_instances() {
#ifdef _DEBUG
    std::cout << "Instantiate C++ objects ..." << std::endl;
#endif
}

void destroy_cpp_instances() {
#ifdef _DEBUG
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
#ifdef _DEBUG
    std::cout << "Set new number of threads:" << threads << std::endl;
    if (!core_list.empty()) {
        std::cout << "Use thread placement: [";
        for (auto it = core_list.begin(); it != core_list.end(); it++) std::cout << *it << " ";
        std::cout << "]";
    }
#endif

    // set worker set size
    global_num_threads = threads;

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

omp_run_until_template = {
    'default':
"""
    run(steps);
    return steps;
""",
    'body':
"""
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
    'single_pop': """
        cond_activated = std::find(populations.begin(), populations.end(), %(id)s) != populations.end();
        if (cond_activated)
            if(or_and)
                stop = stop && pop%(id)s.stop_condition();
            else
                stop = stop || pop%(id)s.stop_condition();
    """
}

omp_initialize_template = """
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

cuda_header_template = """#ifndef __ANNARCHY_H__
#define __ANNARCHY_H__

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <queue>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <cassert>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

/*
 * Built-in functions (host side)
 */
%(built_in)s

/*
 * Custom constants
 *
 */
%(custom_constant)s

/*
 * Custom functions
 * (available on host-side and interfaced for cython)
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

/*
 * Simulation methods
 */
void run(int nbSteps);

int run_until(int steps, std::vector<int> populations, bool or_and);

void step();

/*
 *  Initialization
 */
void initialize(const %(float_prec)s _dt) ;

inline void setDevice(const int device_id) {
#ifdef _DEBUG
    std::cout << "Setting device " << device_id << " as compute device ..." << std::endl;
#endif
    cudaError_t err = cudaSetDevice(device_id);
    if ( err != cudaSuccess )
        std::cerr << "Set device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
}

/*
 *  Life-time management
 */
void create_cpp_instances();
void destroy_cpp_instances();

/*
 * Time export
 */
long int getTime();
void setTime(const long int t_);
%(float_prec)s getDt();
void setDt(const %(float_prec)s dt_);

/*
 * Seed for the RNG (host-side!)
 */
void setSeed(const long int seed, const int num_sources, const bool use_seed_seq);

#endif
"""

cuda_device_kernel = """#include "ANNarchyKernel.cuh"

/********************************************************************/
/*  Device kernel definitions                                       */
/********************************************************************/

/****************************************
 * atomicAdd for non-Pascal             *
 ****************************************/
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                            __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif

/****************************************
 * init random states                   *
 ****************************************/
/*
 *  Each thread gets an unique sequence number (i) and all use the same seed. As highlightet
 *  in section 3.1.1. of the curand documentation this should be enough to get good random numbers
 *
 *  HD(19.7.2019): we need to be careful, that multiple calls to this method need to generate different state sequences.
 */
__global__ void rng_setup_kernel( int N, long long int sequence_offset, curandState* states, unsigned long long seed )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    while( i < N )
    {
        curand_init( seed, i+sequence_offset, 0, &states[ i ] );
        i += blockDim.x * gridDim.x;
    }
}

/****************************************
 * clear psp-related state variables    *
 ****************************************/
__global__ void clear_num_events(unsigned int* num_events) {
    *num_events = 0;
}

__global__ void clear_sum(int num_elem, %(float_prec)s *sum) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    while( j < num_elem ) {
        sum[j] = 0.0;
        j += blockDim.x * gridDim.x;
    }
}

/****************************************
 * common/global available functions    *
 ****************************************/
%(common_kernel)s

/****************************************
 * inline functions                     *
 ****************************************/
%(built_in)s

/****************************************
 * custom constants                     *
 ****************************************/
%(custom_constant)s

/****************************************
 * custom functions                     *
 ****************************************/
%(custom_func)s

/****************************************
 * updating neural variables            *
 ****************************************/
%(pop_kernel)s

/****************************************
 * weighted sum kernels                 *
 ****************************************/
%(psp_kernel)s

/****************************************
 * update synapses kernel               *
 ****************************************/
%(syn_kernel)s

/****************************************
 * global operations kernel             *
 ****************************************/
%(glob_ops_kernel)s

/****************************************
 * postevent kernel                     *
 ****************************************/
%(postevent_kernel)s


/********************************************************************/
/*  Device kernel invocations                                       */
/********************************************************************/

// We need to generate different state sequences per kernel call
static long long int sequence_offset=0;

void init_curand_states( int N, curandState* states, unsigned long long seed ) {
    int numThreads = 64;
    int numBlocks = ceil (float(N) / float(numThreads));

    rng_setup_kernel<<< numBlocks, numThreads >>>( N, sequence_offset, states, seed);
    sequence_offset += N;

#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "init_curand_state: " << cudaGetErrorString(err) << std::endl;
#endif
}

void call_clear_sum(RunConfig cfg, int num_elem, %(float_prec)s *sum) {
    clear_sum<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(num_elem, sum);
}

void call_clear_num_events(RunConfig cfg, unsigned int* num_events) {
    clear_num_events<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(num_events);
}

/****************************************
 * updating neural variables            *
 ****************************************/
 %(pop_invoke_kernel)s

/****************************************
 * weighted sum kernels                 *
 ****************************************/
%(psp_invoke_kernel)s

/****************************************
 * update synapses kernel               *
 ****************************************/
%(syn_invoke_kernel)s

/****************************************
 * global operations kernel             *
 ****************************************/
%(glob_ops_invoke_kernel)s

/****************************************
 * postevent kernel                     *
 ****************************************/
%(postevent_invoke_kernel)s
"""

cuda_device_invoke_header ="""#pragma once
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <iostream>

// Encapsulates the four parameters required for a kernel invocation.
struct RunConfig{
    dim3 nb;
    dim3 tpb;
    int smem_size;
    cudaStream_t stream;

    RunConfig() = default;

    RunConfig(int nb, int tpb, int smem_size, cudaStream_t stream) {
        this->nb = dim3(nb,1,1);
        this->tpb = dim3(tpb,1,1);
        this->smem_size = smem_size;
        this->stream = stream;
    }

    RunConfig(dim3 nb, dim3 tpb, int smem_size, cudaStream_t stream) {
        this->nb = nb;
        this->tpb = tpb;
        this->smem_size = smem_size;
        this->stream = stream;
    }
};

// Pre-defined kernel definitions
void init_curand_states( int N, curandState* states, unsigned long long seed );

void call_clear_sum(RunConfig cfg, int num_elem, %(float_prec)s *sum);
void call_clear_num_events(RunConfig cfg, unsigned int* num_events);

// Model-related kernel definitions
%(invoke_kernel_def)s

"""

cuda_host_body_template =\
"""// ANNarchy-related header
#include "ANNarchy.hpp"
#include "ANNarchyKernel.cuh"

%(prof_include)s
#include <math.h>

// Directly set the number of blocks/number of threads
%(kernel_config)s

// Required by Bell & Garland Kernel
#define MAX_THREADS (30 * 1024)
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

//
// Handling GPU and CPU rng
//

std::vector<std::mt19937> rng;
unsigned long long global_seed;

void setSeed(const long int seed, const int num_sources, const bool use_seed_seq){
    rng.clear();

    if (num_sources == 1) {
        rng.push_back(std::mt19937(seed));
    }else {
        if ( use_seed_seq ) {
            std::seed_seq seq{seed};
            std::vector<std::uint32_t> seeds(num_sources);
            seq.generate(seeds.begin(), seeds.end());

            for (auto it = seeds.begin(); it != seeds.end(); it++) {
                rng.push_back(std::mt19937(*it));
            }
        } else {
            std::cerr << "Not implemented. " << std::endl;
        }
    }

    rng.shrink_to_fit();

    // store the seed for later usage
    global_seed = static_cast<unsigned long long>(seed);
}

/*
 * Internal data
 */
%(float_prec)s dt;
long int t;

// Populations
%(pop_ptr)s

// Projections
%(proj_ptr)s

// Stream configuration (available for CC > 3.x devices)
// NOTE: if the CC is lower then 3.x modification of stream
//       parameter (4th arg) is automatically ignored by CUDA
%(stream_setup)s

// Initialize the internal data
void initialize(%(float_prec)s _dt) {
%(initialize)s
}

/*
 *  Life-time management
 */
void create_cpp_instances() {
#ifdef _DEBUG
    std::cout << "Instantiate C++ objects ..." << std::endl;
#endif
}

void destroy_cpp_instances() {
#ifdef _DEBUG
    std::cout << "Destroy C++ objects ..." << std::endl;
#endif
}

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
            delete recorders[i];
            recorders[i] = nullptr;
            break;
        }
    }
}

/**
 *  Implementation remark (27.02.2015, HD) to: run(int), step() and single_step()
 *
 *  we have two functions in ANNarchy to run simulation code: run(int) and step(). The latter one to
 *  run several steps at once, the other one just a single step. On CUDA I face the problem, that I
 *  propably need to update variables before step() and definitly changed variables after step().
 *  run(int) calls step() normally, if I add transfer codes in step(), run(N) would end up in N
 *  back transfers from GPUs, whereas we only need the last one.
 *  As solution I renamed step() to single_step(), the interface behaves as OMP side and I only
 *  transfer at begin and at end, as planned.
 */

// Step method. Generated by ANNarchy. (analog to step() in OMP)
void single_step()
{
%(prof_step_pre)s

%(prof_proj_psp_pre)s
    ////////////////////////////////
    // Clear sums
    ////////////////////////////////
%(clear_sums)s

    ////////////////////////////////
    // Presynaptic events
    ////////////////////////////////
%(compute_sums)s

%(prof_proj_psp_post)s

    ////////////////////////////////
    // Recording targets
    ////////////////////////////////
    for(int i=0; i < recorders.size(); i++){
        if (recorders[i])
            recorders[i]->record_targets();
    }

    ////////////////////////////////
    // Update neural variables
    ////////////////////////////////
%(prof_neur_step_pre)s
%(update_neuron)s

%(prof_neur_step_post)s

%(update_FR)s

    ////////////////////////////////
    // Delay outputs
    ////////////////////////////////
%(delay_code)s

    ////////////////////////////////
    // Global operations (min/max/mean)
    ////////////////////////////////
%(update_globalops)s

    ////////////////////////////////
    // Update synaptic variables
    ////////////////////////////////
%(prof_proj_step_pre)s
%(update_synapse)s

%(prof_proj_step_post)s

    ////////////////////////////////
    // Postsynaptic events
    ////////////////////////////////
%(post_event)s

    ////////////////////////////////
    // Recording neural/synaptic variables
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
    t++;    // host side, provided as argument to kernels

%(prof_step_post)s
}

// Simulate the network for the given number of steps
void run(const int nbSteps) {
#ifdef _DEBUG
    std::cout << "host to device transfers." << std::endl;
#endif

%(host_device_transfer)s

    stream_assign();

#ifdef _DEBUG
    std::cout << "simulate " << nbSteps << " steps." << std::endl;
#endif

%(prof_run_pre)s
    // simulation loop
    for(int i=0; i<nbSteps; i++) {
        single_step();
    }
%(prof_run_post)s

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error occured during simulation: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}

int run_until(const int steps, std::vector<int> populations, bool or_and) {
%(run_until)s
}

void step() {
%(host_device_transfer)s
%(prof_run_pre)s
    single_step();
%(prof_run_post)s

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "An error occured within simulation loop: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}

/*
 * Access to time and dt
 *
 */
long int getTime() {return t;}
void setTime(const long int t_) { t=t_; }
%(float_prec)s getDt() { return dt;}
void setDt(const %(float_prec)s dt_) { dt=dt_;}
"""

cuda_stream_setup="""
cudaStream_t streams[%(nbStreams)s];

void stream_setup()
{
    for ( int i = 0; i < %(nbStreams)s; i ++ )
    {
        cudaStreamCreate( &streams[i] );
    }
}

void stream_assign()
{
%(pop_assign)s

%(proj_assign)s
}

void stream_destroy()
{
    for ( int i = 0; i < %(nbStreams)s; i ++ )
    {
        // all work finished
        cudaStreamSynchronize( streams[i] );

        // destroy
        cudaStreamDestroy( streams[i] );
    }
}
"""

cuda_initialize_template = """
    dt = _dt;
    t = (long int)(0);

%(prof_init)s

%(pop_init)s

%(proj_init)s

    // global constants
%(custom_constant)s

    // create streams
    stream_setup();
"""

built_in_functions = """
#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) % long(b)
#define ite(a, b, c) (a?b:c)
"""

integer_power_cpu="""
// power function for integer exponent
inline %(float_prec)s power(%(float_prec)s x, unsigned int a){
    %(float_prec)s res=x;
    for (unsigned int i=0; i< a-1; i++){
        res *= x;
    }
    return res;
};
"""

integer_power_cuda="""
// power function for integer exponent
__device__ %(float_prec)s power(%(float_prec)s x, unsigned int a) {
    %(float_prec)s res=x;
    for (unsigned int i = 0; i < a-1; i++){
        res *= x;
    }
    return res;
}
"""
