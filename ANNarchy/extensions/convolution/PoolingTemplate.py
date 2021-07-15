# =============================================================================
#
#     PoolingTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2020  Julien Vitay <julien.vitay@gmail.com>,
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
pooling_template_omp = {
    'include_additional': '#include <limits>',

    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    // connectivity data
    std::vector<int> post_rank;
    std::vector< std::vector<int> > pre_rank;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses() { 
        int size = 0;
        for(auto it = pre_rank.cbegin(); it != pre_rank.cend(); it++)
            size += it->size(); 
        return size;
    }
    int dendrite_size(int n) { return pre_rank[n].size(); }
    int nb_dendrites() { return post_rank.size(); }
""",

    # Export the connectivity matrix
    'export_connectivity': """
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        int nb_synapses()
        int dendrite_size(int n)
        int nb_dendrites()
""",

    # No additional variables
    'declare_parameters_variables': "",
    'access_parameters_variables': "",
    'export_parameters_variables': "",

    # Arguments to the wrapper constructor
    'wrapper_args': "weights, coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_rank(coords)
""",
    # Something like init_from_lil?
    'wrapper_connector_call': "",

    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()
    def nb_synapses(self):
        return proj%(id_proj)s.nb_synapses()
    def dendrite_size(self, lil_idx):
        return proj%(id_proj)s.dendrite_size(lil_idx)
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables': "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=%(sum_default)s;
    """,

    # Delays
    'wrapper_init_delay': "",
    
    # Override the monitor to avoid recording the weights
    'monitor_class':"""
""",
    'monitor_export': """
""",
     'monitor_wrapper': """
"""
}

pooling_template_cuda = {
    'include_additional': '#include <cfloat>',

    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    std::vector< std::vector<int> > coords;
    int *gpu_coords;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to pre-synaptic coordinates (upper left corner)
    std::vector< std::vector<int> > get_coords() { return this->coords; }
    void set_coords(std::vector< std::vector<int> > coords) {
        this->coords = coords;

        // Flattening coords
        auto num_coords = coords.size();
        auto coord_width = coords[0].size();
        auto flat_coords = std::vector<int>(num_coords*coord_width, 0);
        for (auto i = 0; i < num_coords; i++) {
            for( auto j = 0; j < coord_width; j++ ) {
                flat_coords[i*coord_width+j] = coords[i][j];
            }
        }

        cudaMalloc((void**)&gpu_coords, flat_coords.size()*sizeof(int));
        cudaMemcpy( gpu_coords, flat_coords.data(), flat_coords.size()*sizeof(int), cudaMemcpyHostToDevice);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Pooling: " << cudaGetErrorString(err) << std::endl;
        }
    }
    int dendrite_size(int n) { return 0; }
    int nb_synapses() { return 0; }
""",

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[vector[int]] get_coords()
        void set_coords(vector[vector[int]])
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "weights, coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_coords(coords)
""",
    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return np.arange(0, %(size_post)s)
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_coords()
    def nb_synapses(self):
        return proj%(id_proj)s.nb_synapses()
    def dendrite_size(self, n):
        return proj%(id_proj)s.dendrite_size(n)
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables': "",

    'init_connectivity_matrix': "",

    # Memory transfer of variables
    'host_device_transfer': "",
    'device_host_transfer': "",

    # Override the monitor to avoid recording the weights
    'monitor_class':"",
    'monitor_export': "",
    'monitor_wrapper': ""
}

cuda_op_code = {
    'min': """if ( local_r < local_res ) local_res = local_r;""",
    'max': """if ( local_r > local_res ) local_res = local_r;""",
    'mean': "local_res += local_r;"
}

#
# For really small kernels it turns out to be beneficial
# to perform the operation with a single thread per block.
cuda_pooling_code_2d_small_extent = {
    'psp_body': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int num_centers, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r) {
    int bIdx = blockIdx.x;
    int tid = threadIdx.x;

    // shared storage need to be initialized
    extern %(float_prec)s __shared__ sdata[];
    sdata[tid] = %(sum_default)s;
    __syncthreads();

    int coord_idx = floor(tid / static_cast<%(float_prec)s>(%(col_extent)s)) + blockIdx.x * floor( blockDim.x / static_cast<%(float_prec)s>(%(col_extent)s) );
    if (coord_idx < num_centers) {
        int x_coords = centers[2*coord_idx];
        int y_coords = centers[2*coord_idx+1];

        int idx_x, idx_y;
        %(float_prec)s local_r = 0.0;
        %(float_prec)s local_res = %(sum_default)s;

        for( int x = 0; x < %(row_extent)s; x++ ) {
            idx_x = x_coords + x;
            if ( idx_x >= %(row_size)s )
                continue;

            for( int y = 0; y < %(col_extent)s; y++ ) {
                idx_y = y_coords + y;
                if ( idx_y >= %(col_size)s )
                    continue;

                local_r = r [ idx_x * %(col_size)s + idx_y ];
%(operation)s
            }
        }

        // store intermediate result
        sdata[tid] = local_res;
        __syncthreads();

        // Reduction in shared memory
%(operation_reduce)s
    }
};
""",
    'psp_header': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int num_centers, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r);
""",
    'psp_call': """
    int coords_per_block = floor(32.0 / static_cast<%(float_prec)s>(%(col_extent)s));
    int num_blocks = ceil(static_cast<%(float_prec)s>(%(size_post)s) / static_cast<%(float_prec)s>(coords_per_block));
    int thread_per_block = %(col_extent)s * coords_per_block;
    int shared_mem_size = thread_per_block * sizeof(%(float_prec)s);
    pooling_proj%(id_proj)s<<< num_blocks, thread_per_block, thread_per_block >>>( pop%(id_post)s.gpu__sum_%(target)s, %(size_post)s, proj%(id_proj)s.gpu_coords, pop%(id_pre)s.gpu_r );
""",
    # The reduction stage is responsible to fuse the several local results within
    # the warp to the final result. ATTENTION: there are several results in this warp
    'reduce_code': {
        'max': """
    if ( tid %% %(col_extent)s == 0 ) {
        for(int y = 1; y < %(col_extent)s; y++)
            sdata[tid] = max(sdata[tid], sdata[tid+y]);

        psp[coord_idx] = sdata[tid];
    }
""",
    }
}

#
# Pooling implementation where a warp handles a row
# at once.
cuda_pooling_code_2d = {
    'psp_body': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int shared_size, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r) {
    int bIdx = blockIdx.x;
    int tid = threadIdx.x;

    // get pre-synaptic coordinates
    int x_coords = centers[2*bIdx];
    int y_coords = centers[2*bIdx+1];

    // shared storage need to be initialized
    extern %(float_prec)s __shared__ sdata[];
    for(int x = tid; x < shared_size; x++)
        sdata[x] = %(sum_default)s;
    __syncthreads();

    // local variables
    int idx_x, idx_y;
    %(float_prec)s local_r = 0.0;
    %(float_prec)s local_res = %(sum_default)s;

    if (tid < %(col_extent)s) {

        // row-wise scan
        for( int x = 0; x < %(row_extent)s; x++ ) {
            idx_x = x_coords + x;
            if ( idx_x >= %(row_size)s )
                continue;

            for( int y = tid; y < %(col_extent)s; y += blockDim.x ) {
                idx_y = y_coords + y;
                if ( idx_y >= %(col_size)s )
                    continue;

                local_r = r [ idx_x * %(col_size)s + idx_y ];
%(operation)s
            }
        }

        // store intermediate result
        sdata[tid] = local_res;
        __syncthreads();

        // Reduction in shared memory
%(operation_reduce)s
    }
}
""",
    'psp_header': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int shared_size, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r);
""",
    # Technically, we could use more than warp-size threads.
    # But as we often have small extents it is not required (HD: 27. Nov. 2020)
    'psp_call': """
    auto tpb = 32;
    auto shared_size = min(32, tpb);
    auto smem_size = 2 * shared_size * sizeof(%(float_prec)s);
    pooling_proj%(id_proj)s<<< %(size_post)s, tpb, smem_size >>>( pop%(id_post)s.gpu__sum_%(target)s, 2*shared_size, proj%(id_proj)s.gpu_coords, pop%(id_pre)s.gpu_r );
""",
    # The reduction stage is responsible to fuse the
    # several local results within the warp to the final result
    'reduce_code': {
        'min': """if (tid < 16)
{
    volatile %(float_prec)s* smem = sdata;

    smem[tid] = min(smem[tid], smem[tid + 16]);
    smem[tid] = min(smem[tid], smem[tid + 8]);
    smem[tid] = min(smem[tid], smem[tid + 4]);
    smem[tid] = min(smem[tid], smem[tid + 2]);
    smem[tid] = min(smem[tid], smem[tid + 1]);
}
__syncthreads();

// write result for this block to global mem
if (tid == 0)
{
    psp[bIdx] = sdata[0];
}
""",
        'max': """if (tid < 16)
{
    volatile %(float_prec)s* smem = sdata;

    smem[tid] = max(smem[tid], smem[tid + 16]);
    smem[tid] = max(smem[tid], smem[tid + 8]);
    smem[tid] = max(smem[tid], smem[tid + 4]);
    smem[tid] = max(smem[tid], smem[tid + 2]);
    smem[tid] = max(smem[tid], smem[tid + 1]);
}
__syncthreads();

// write result for this block to global mem
if (tid == 0)
{
    psp[bIdx] = sdata[0];
}
""",
        'mean': """if (tid < 16)
{
    volatile %(float_prec)s* smem = sdata;

    smem[tid] = smem[tid] + smem[tid + 16];
    smem[tid] = smem[tid] + smem[tid + 8];
    smem[tid] = smem[tid] + smem[tid + 4];
    smem[tid] = smem[tid] + smem[tid + 2];
    smem[tid] = smem[tid] + smem[tid + 1];
}
__syncthreads();

// write result for this block to global mem
if (tid == 0)
{
    psp[bIdx] = sdata[0] / static_cast<%(float_prec)s>(%(row_extent)s * %(col_extent)s);
}
"""
    }
}

cuda_pooling_code_3d = {
    'psp_body': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int shared_size, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r) {
    int bIdx = blockIdx.x;
    int tid = threadIdx.x;

    int x_coords = centers[3*bIdx];
    int y_coords = centers[3*bIdx+1];
    int z_coords = centers[3*bIdx+2];

    int idx_x, idx_y, idx_z;
    %(float_prec)s local_r = 0.0;
    %(float_prec)s local_res = %(sum_default)s;

    for( int x = 0; x < %(row_extent)s; x++ ) {
        idx_x = x_coords + x;
        if ( idx_x >= %(row_size)s )
            continue;

        for( int y = 0; y < %(col_extent)s; y++ ) {
            idx_y = y_coords + y;
            if ( idx_y >= %(col_size)s )
                continue;

            for(int z = 0; z < %(plane_extent)s; z++) {
                idx_z = z_coords + z;
                if ( idx_z >= %(plane_size)s )
                    continue;

                local_r = r [ %(plane_size)s * (%(col_size)s * idx_x + idx_y) + idx_z ];
%(operation)s
            }
        }
    }

    psp[bIdx] = local_res;
""",
    'psp_header': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* __restrict__ psp, const int* __restrict__ centers, const %(float_prec)s* __restrict__ r);
""",
    'psp_call': """
    pooling_proj%(id_proj)s<<< %(size_post)s, 1 >>>( pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_coords, pop%(id_pre)s.gpu_r );
"""
}
