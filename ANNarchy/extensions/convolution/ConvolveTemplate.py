# =============================================================================
#
#     ConvolutionTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2019-2020  Julien Vitay <julien.vitay@gmail.com>,
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
# =============================================================================

convolve_template_omp = {
    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    // Connectivity data
    std::vector< std::vector<int> > pre_coords;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<std::vector<int>> get_pre_coords() { return pre_coords; }
    void set_pre_coords(std::vector<std::vector<int>> coords) { pre_coords = coords; }
""" ,

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[vector[int]] get_pre_coords()
        void set_pre_coords(vector[vector[int]])
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_pre_coords(coords)
""",

    # Delays
    'wrapper_init_delay': "",

    # Something like init_from_lil?
    'wrapper_connector_call': "",

    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    property size:
        def __get__(self):
            return %(size_post)s
    def post_rank(self):
        return list(np.arange(0, %(size_post)s))
    def pre_coords(self):
        return proj%(id_proj)s.get_pre_coords()
    def nb_synapses(self):
        return 0
    def dendrite_size(self, lil_idx):
        return 0
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables' : "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;
""",

    # Memory Management
    'clear': """
        // pre-coords
        for (auto it = pre_coords.begin(); it != pre_coords.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        pre_coords.clear();
        pre_coords.shrink_to_fit();
""",

    'size_in_bytes': """
        // pre-coords
        size_in_bytes += sizeof(std::vector<std::vector<int>>);
        size_in_bytes += pre_coords.capacity() * sizeof(std::vector<int>);
        for (auto it = pre_coords.begin(); it != pre_coords.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(int);
        }
"""
}

convolve_template_cuda = {
    'include_additional': '#include "VecTransformation.hpp"',

    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    // Connectivity data
    std::vector< std::vector<int> > pre_coords;
    int* gpu_pre_coords;
    bool pre_coords_dirty;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<std::vector<int>> get_pre_coords() { return pre_coords; }
    void set_pre_coords(std::vector<std::vector<int>> coords) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::set_pre_coords()"<< std::endl;
    #endif
        pre_coords = coords;
        pre_coords_dirty = true;
    }
""" ,

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[vector[int]] get_pre_coords()
        void set_pre_coords(vector[vector[int]])
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_pre_coords(coords)
""",

    # This template concerns only the connectivity where
    # no read-back is required
    "device_host_transfer": "",
    "host_device_transfer": """
        if (pre_coords_dirty) {
        #ifdef _DEBUG
            std::cout << "ProjStruct%(id_proj)s (convolution): update device coords." << std::endl;
        #endif
            auto flat_coords = transform_2d_to_1d<int>(pre_coords);
            size_t size_in_bytes = flat_coords.size() * sizeof(int);

            cudaMalloc((void**)&gpu_pre_coords, size_in_bytes);
            cudaMemcpy(gpu_pre_coords, flat_coords.data(), size_in_bytes, cudaMemcpyHostToDevice);
        }
""",

    # Delays
    'wrapper_init_delay': "",

    # Something like init_from_lil?
    'wrapper_connector_call': "",

    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    property size:
        def __get__(self):
            return %(size_post)s
    def post_rank(self):
        return list(np.arange(0, %(size_post)s))
    def pre_coords(self):
        return proj%(id_proj)s.get_pre_coords()
    def nb_synapses(self):
        return 0
    def dendrite_size(self, lil_idx):
        return 0
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables' : "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;
""",

    # Memory Management
    'clear': "",
    'size_in_bytes': ""
}

conv_filter_template = {
    # The Python extension definitions are the same
    # for single-thread/OpenMP and CUDA
    "pyx_wrapper": {
        "args": ", weights",
        "export": """
        # Local variable w
        %(type_w)s get_w()
        void set_w(%(type_w)s)
""",
        "init": """
        proj%(id_proj)s.set_w(weights)
""",
        "access": """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def set_w(self, value):
        proj%(id_proj)s.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_w()
    def set_dendrite_w(self, int rank, value):
        proj%(id_proj)s.set_w(value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return 0.0
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
        pass
"""
    },
    # Single-thread/OpenMP
    "openmp": {
        "access": """
    // Local parameter w
    %(type_w)s get_w() { return w; }
    void set_w(%(type_w)s value) { w = value; }
""",
    },
    # CUDA
    "cuda": {
        "declare": """\t// Filter definition
    %(cpu_side_filter)s
    %(float_prec)s *gpu_w;
    bool host_w_dirty;
""",
        "access": """
    // Local parameter w
    %(type_w)s get_w() { return w; }
    void set_w(%(type_w)s value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s (convolution): set new filter on host" << std::endl;
    #endif
        w = value;
        host_w_dirty = true;
    }
""",
    'host_device_transfer': """
        if ( host_w_dirty ) {
        #ifdef _DEBUG
            std::cout << "ProjStruct%(id_proj)s (convolution): update device filter." << std::endl;
        #endif
            auto flat_data = transform_%(pre_dim)sd_to_1d<%(ctype)s>(w);
            auto size_in_bytes = flat_data.size() * sizeof(%(ctype)s);

            cudaMalloc((void**)&gpu_w, size_in_bytes);
            auto malloc_err = cudaGetLastError();
            if (malloc_err != cudaSuccess) {
                std::cerr << "ProjStruct%(id_proj)s::host_to_device - cudaMalloc: " << cudaGetErrorString(malloc_err) << std::endl;
            }

            cudaMemcpy(gpu_w, flat_data.data(), size_in_bytes, cudaMemcpyHostToDevice);
            auto memcpy_err = cudaGetLastError();
            if (memcpy_err != cudaSuccess) {
                std::cerr << "ProjStruct%(id_proj)s::host_to_device - cudaMemcpy: " << cudaGetErrorString(memcpy_err) << std::endl;
            }

            host_w_dirty = false;
        }
""",
    # No read-back needed, as weights are not changed during simulation
    'device_host_transfer': ""
    }
}

cuda_convolution_single_filter = {
    "body": """__global__ void convolution_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    %(float_prec)s sum;
    int rk_pre, w_idx;
    const int *coord = &pre_coords[%(pre_dim)s*bIdx];

%(convolve_code)s

    psp[bIdx] += sum;
}
""",
    "header": "__global__ void convolution_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active )
        convolution_proj%(id_proj)s<<<pop%(id_post)s.size, 1>>>(pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s);
"""
}

cuda_convolution_bank_of_filter = {
    "body": """__global__ void convolution_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    %(float_prec)s sum;
    int rk_pre, w_idx;
    const int *coord = &pre_coords[%(filter_dim)s*bIdx];
    const %(float_prec)s *w_bank = &w[coord[%(pre_dim)s] * %(num_elem_filter)s];

%(convolve_code)s

    psp[bIdx] += sum;
}
""",
    "header": "__global__ void convolution_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active )
        convolution_proj%(id_proj)s<<<pop%(id_post)s.size, 1>>>(pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s);
"""
}
