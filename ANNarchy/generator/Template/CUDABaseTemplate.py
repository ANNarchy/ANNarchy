"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

header_template = """#ifndef __ANNARCHY_H__
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

// Useful functions
#include "helper_functions.cuh"

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
size_t estimate_record_size(int num_steps);

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

device_invoke_header = """#pragma once
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
void init_curand_states( int numBlocks, int numThreads, curandState* states, unsigned long long seed );

void call_clear_sum(RunConfig cfg, int num_elem, %(float_prec)s *sum);
void call_clear_num_events(RunConfig cfg, unsigned int* num_events);

// Model-related kernel definitions
%(invoke_kernel_def)s
"""

device_kernel = """#include "ANNarchyKernel.cuh"

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
 *  Each thread gets an unique sequence number (tid) and all use the same seed. As highlighted
 *  in section 3.1.1. of the curand documentation this should be enough to get good random numbers
 *
 *  HD (19.7.2019):     we need to be careful, that multiple calls to this method need to generate different state sequences.
 *  HD (17.10.2025):    Note, ANN5.0 switches from per-element RNG state to per-thread RNG state!
 */
__global__ void rng_setup_kernel( int num_total, long long int sequence_offset, curandState* states, unsigned long long seed )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_total)
    {
        curand_init( seed, tid+sequence_offset, 0, &states[ tid ] );
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

void init_curand_states( int numBlocks, int numThreads, curandState* states, unsigned long long seed ) {

    rng_setup_kernel<<< numBlocks, numThreads >>>( numBlocks * numThreads, sequence_offset, states, seed);
    sequence_offset += numBlocks * numThreads;

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

host_body_template = """// ANNarchy-related header
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
            // mark position as free
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
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- evaluate step " << t << " (" << t * dt << " ms) --" << std::endl;
    std::cout << std::flush;
#endif

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

#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "-- simulation step " << t << " completed --" << std::endl;
    std::cout << std::flush;
#endif
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

%(device_host_transfer)s

    cudaDeviceSynchronize();
}

int run_until(const int steps, std::vector<int> populations, bool or_and) {
%(run_until)s
}

void step() {
#ifdef _DEBUG
    std::cout << "host to device transfers." << std::endl;
#endif
%(host_device_transfer)s

    stream_assign();

#ifdef _DEBUG
    std::cout << "simulate a single step." << std::endl;
#endif

%(prof_run_pre)s
    single_step();
%(prof_run_post)s

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "An error occured within simulation loop: " << cudaGetErrorString(err) << std::endl;
    }

%(device_host_transfer)s

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

host_initialize_template = """
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
    """
}

stream_setup = """
cudaStream_t streams[%(nbStreams)s];

void stream_setup()
{
    for ( int i = 0; i < %(nbStreams)s; i ++ )
    {
        cudaStreamCreate( &streams[i] );
    }
    auto err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cerr << "Error occured during stream_setup(): " << cudaGetErrorString(err) << std::endl;
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

    auto err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cerr << "Error occured during stream_destroy(): " << cudaGetErrorString(err) << std::endl;
}
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
__device__ %(float_prec)s power(%(float_prec)s x, unsigned int a) {
    %(float_prec)s res=x;
    for (unsigned int i = 0; i < a-1; i++){
        res *= x;
    }
    return res;
}
"""
