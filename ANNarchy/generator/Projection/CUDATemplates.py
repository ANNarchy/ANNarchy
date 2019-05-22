#===============================================================================
#
#     CUDATemplates.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
projection_header = """#pragma once

#include "pop%(id_pre)s.hpp"
#include "pop%(id_post)s.hpp"
%(include_additional)s
%(include_profile)s

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;
%(struct_additional)s

/////////////////////////////////////////////////////////////////////////////
// proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;

%(declare_connectivity_matrix)s
%(declare_inverse_connectivity_matrix)s
%(declare_delay)s
%(declare_event_driven)s
%(declare_rng)s
%(declare_parameters_variables)s
%(declare_cuda_stream)s
%(declare_additional)s
%(declare_profile)s

    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

%(init_connectivity_matrix)s

        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();

%(init_event_driven)s
%(init_parameters_variables)s
%(init_rng)s
%(init_additional)s
%(init_profile)s
    }

    // Spiking networks: inverse the connectivity matrix
    void inverse_connectivity_matrix() {
%(init_inverse_connectivity_matrix)s
    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Additional access methods
%(access_connectivity_matrix)s
%(access_parameters_variables)s
%(access_additional)s

    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(determine_size)s
        return size_in_bytes;
    }

    void clear() {
%(clear_container)s
    }

    // Memory transfers
    void host_to_device() {
%(host_to_device)s
    }

    void device_to_host() {
%(device_to_host)s
    }

%(cuda_flattening)s
};
"""

# Definition for the usage of CUDA device random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
curand = {
    'local': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, overallSynapses * sizeof(curandState));
        init_curand_states( overallSynapses, gpu_%(rd_name)s, seed );
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "proj%(id)s - init_projection: " << cudaGetErrorString(err) << std::endl;
    #endif

"""
    },
    'global': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, size * sizeof(curandState));
        init_curand_states( size, gpu_%(rd_name)s, seed );
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "proj%(id)s - init_projection: " << cudaGetErrorString(err) << std::endl;
    #endif
"""
    }
}

cuda_stream = """
    // stream
    cudaStream_t stream;
"""

#
# Set of functions for convertion of LIL in CSR
cuda_flattening = """
    /*
     * (De-)Flattening of LIL structures
     */
    void genRowPtr( ) {
        std::vector<std::vector<int> >::iterator pre_it = pre_rank.begin();
        std::vector<int>::iterator post_it = post_rank.begin();
        row_ptr = std::vector<int>(pop%(id_post)s.size, 0);
        int curr_off = 0;
        for(int i = 0; i < pop%(id_post)s.size; i++) {
            row_ptr[i] = curr_off;
            if ( i == *post_it ) {
                curr_off += pre_it->size();
                pre_it++;
                post_it++;
            }
        }
        row_ptr.push_back(curr_off);
        overallSynapses = curr_off;
    }

    template<typename T>
    std::vector<T> flattenArray(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatVec = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        for ( it = in.begin(); it != in.end(); it++)
        {
            flatVec.insert(flatVec.end(), it->begin(), it->end());
        }

        return flatVec;
    }

    template<typename T>
    std::vector<std::vector<T> > deFlattenArray( std::vector<T> in )
    {
        std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();
        std::vector<int>::iterator it;

        int t=0;
        for ( int i = 0; i < pop%(id_post)s.size; i++)
        {
            if ( row_ptr[i] != row_ptr[i+1] ) {
                int num_syn = row_ptr[i+1]-row_ptr[i];
                std::vector<T> tmp = std::vector<T>(in.begin()+t, in.begin()+t+num_syn);
                t += num_syn;

                deFlatVec.push_back(tmp);
            }
        }

        return deFlatVec;
    }

    template<typename T>
    std::vector<T> deFlattenDendrite ( std::vector<T> in, int rank )
    {
        std::vector<T> deFlatVec = std::vector<T>();
        std::vector<int>::iterator it;

        if ( row_ptr[rank] != row_ptr[rank+1] ) {
            deFlatVec = std::vector<T>(in.begin()+row_ptr[rank], in.begin()+row_ptr[rank+1]);
        }

        return deFlatVec;
    }
"""

# some base stuff
cuda_templates = {
    'projection_header': projection_header,
    'rng': curand,
    'cuda_stream': cuda_stream,
    'flattening': cuda_flattening,
}
