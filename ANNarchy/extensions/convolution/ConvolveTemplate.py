"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

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
    'clear': """
        // pre-coords
        for (auto it = pre_coords.begin(); it != pre_coords.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        pre_coords.clear();
        pre_coords.shrink_to_fit();

        cudaFree(gpu_pre_coords);
""",
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
    "body": """__global__ void cu_conv_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    %(float_prec)s sum;
    int rk_pre, w_idx;
    const int *coord = &pre_coords[%(pre_dim)s*bIdx];

%(convolve_code)s

    psp[bIdx] += sum;
}
""",
    "invoke": """
void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    cu_conv_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(psp, pre_coords, w%(pre_variables_invoke)s);
}
""",
    "header": "void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active ) {
        convolution_proj%(id_proj)s(
            RunConfig(pop%(id_post)s.size, 1, 0, proj%(id_proj)s.stream),
            pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s
        );
    
    #ifdef _DEBUG
        auto proj%(id_proj)s_conv_err = cudaDeviceSynchronize();
        if ( proj%(id_proj)s_conv_err != cudaSuccess) {
            std::cout << "Convolution projection %(id_proj)s - psp: " << cudaGetErrorString( proj%(id_proj)s_conv_err ) << std::endl;
        }
    #endif
    }
"""
}

cuda_convolution_bank_of_filter = {
    "body": """__global__ void cu_conv_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    %(float_prec)s sum;
    int rk_pre, w_idx;
    const int *coord = &pre_coords[%(filter_dim)s*bIdx];
    const %(float_prec)s *w_bank = &w[coord[%(pre_dim)s] * %(num_elem_filter)s];

%(convolve_code)s

    psp[bIdx] += sum;
}
""",
    "invoke": """void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s) {
    cu_conv_proj%(id_projs)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream(psp, pre_coords, w%(pre_variables_invoke)s);
}
""",
    "header": "void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active ) {
        convolution_proj%(id_proj)s(
            RunConfig(pop%(id_post)s.size, 1, 0, proj%(id_proj)s.stream),
            pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s
        );

    #ifdef _DEBUG
        auto proj%(id_proj)s_conv_err = cudaDeviceSynchronize();
        if ( proj%(id_proj)s_conv_err != cudaSuccess) {
            std::cout << "Convolution projection %(id_proj)s - psp: " << cudaGetErrorString( proj%(id_proj)s_conv_err ) << std::endl;
        }
    #endif
    }
"""
}

# Specialized code template: 3D filter bank, 2D pre-population
cuda_convolution_bank_of_filter_3d = {
    "body": """__global__ void cu_conv_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    int i_w = threadIdx.x;
    int j_w = threadIdx.y;
    int w_idx = %(filter_dim_j)s*(i_w) + j_w;

    const int *coord = &pre_coords[3*bIdx];
    const double *w_bank = &w[coord[2] * %(num_elem_filter)s];

    int i_pre = coord[0] + (i_w - %(pre_offset_i)s);
    if ((i_pre < 0) || (i_pre > %(pre_border_i)s)) {
        return;
    }

    int j_pre = coord[1] + (j_w - %(pre_offset_j)s);
    if ((j_pre < 0) || (j_pre > %(pre_border_j)s)) {
        return;
    }

    int rk_pre = %(pre_dim_j)s*(i_pre) + j_pre;
    atomicAdd(&psp[bIdx], %(pre_variable)s[rk_pre]*w_bank[w_idx]);
}
""",
    "invoke": """
void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s) {
    cu_conv_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(psp, pre_coords, w%(pre_variables_invoke)s);
}
""",
    "header": "void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active ) {
        auto num_blocks = dim3(pop%(id_post)s.size, 1, 1);
        auto thread_config = dim3(%(filter_dim_i)s, %(filter_dim_j)s, 1);
        convolution_proj%(id_proj)s(
            RunConfig(num_blocks, thread_config, 0, proj%(id_proj)s.stream),
            pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s
        );

    #ifdef _DEBUG
        auto proj%(id_proj)s_conv_err = cudaDeviceSynchronize();
        if ( proj%(id_proj)s_conv_err != cudaSuccess) {
            std::cout << "Convolution projection %(id_proj)s - psp: " << cudaGetErrorString( proj%(id_proj)s_conv_err ) << std::endl;
        }
    #endif        
    }
"""
}

cuda_convolution_bank_of_filter_4d = {
    "body": """__global__ void cu_conv_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_coords, const %(float_prec)s* __restrict__ w%(pre_variables_header)s) {
    int bIdx = blockIdx.x;
    int i_w = threadIdx.x;
    int j_w = threadIdx.y;
    int rk_pre, w_idx;

    const int *coord = &pre_coords[4*bIdx];
    const double *w_bank = &w[coord[3] * %(num_elem_filter)s];

    int i_pre = coord[0] + (i_w - %(pre_offset_i)s);
    if ((i_pre < 0) || (i_pre > %(pre_border_i)s)) {
        return;
    }

    int j_pre = coord[1] + (j_w - %(pre_offset_j)s);
    if ((j_pre < 0) || (j_pre > %(pre_border_j)s)) {
        return;
    }

    %(float_prec)s sum = 0.0;
    for (int k_w = 0; k_w < 64; k_w++) {
        int k_pre = coord[2] + (k_w - %(pre_offset_k)s);
        if ((k_pre < 0) || (k_pre > %(pre_border_k)s)) {
            continue;
        }

        rk_pre = %(pre_dim_k)s * ( %(pre_dim_j)s*(i_pre) + j_pre ) + k_pre;
        w_idx = %(filter_dim_k)s*(%(filter_dim_j)s*(i_w) + j_w) + k_w;
        sum += %(pre_variable)s[rk_pre]*w_bank[w_idx];
    }

    atomicAdd(&psp[bIdx], sum);
}
""",
    "invoke": """
void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s) {
    cu_conv_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(psp, pre_coords, w%(pre_variables_invoke)s);
}
""",
    "header": "void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_coords, const %(float_prec)s* w%(pre_variables_header)s);",
    "call": """
    if (proj%(id_proj)s._transmission && pop%(id_post)s._active ) {
        auto num_blocks = dim3(pop%(id_post)s.size, 1, 1);
        auto thread_config = dim3(%(filter_dim_i)s, %(filter_dim_j)s, 1);
        convolution_proj%(id_proj)s(
            RunConfig(num_blocks, thread_config, 0, proj%(id_proj)s.stream),
            pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_pre_coords, proj%(id_proj)s.gpu_w%(pre_variables_call)s
        );

    #ifdef _DEBUG
        auto proj%(id_proj)s_conv_err = cudaDeviceSynchronize();
        if ( proj%(id_proj)s_conv_err != cudaSuccess) {
            std::cout << "Convolution projection %(id_proj)s - psp: " << cudaGetErrorString( proj%(id_proj)s_conv_err ) << std::endl;
        }
    #endif
    }
"""
}

