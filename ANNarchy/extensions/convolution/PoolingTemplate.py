# =============================================================================
#
#     PoolingTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2019  Julien Vitay <julien.vitay@gmail.com>,
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
    int nb_synapses(int n) { return pre_rank[n].size(); }
""",

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "weights, coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_rank(coords)
""",
    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()
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
        auto flat_coords = flattenArray< int >(coords);
        cudaMalloc((void**)&gpu_coords, flat_coords.size()*sizeof(int));
        cudaMemcpy( gpu_coords, flat_coords.data(), flat_coords.size()*sizeof(int), cudaMemcpyHostToDevice);
    }
    int nb_synapses(int n) { return 0; }
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
    def nb_synapses(self, n):
        return proj%(id_proj)s.nb_synapses(n)
""",

    # Wrapper access to variables
    'wrapper_access_parameters_variables': "",

    # Variables for the psp code
    'psp_header': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* psp, int* centers, %(float_prec)s* r);
    """,
    'psp_body': """__global__ void pooling_proj%(id_proj)s ( %(float_prec)s* psp, int* centers, %(float_prec)s* r) {
    int bIdx = blockIdx.x;

    %(pooling_code)s
    psp[bIdx] = local_res;
};
    """,
    'psp_call': """\tpooling_proj%(id_proj)s<<< %(size_post)s, 1 >>>( pop%(id_post)s.gpu__sum_%(target)s, proj%(id_proj)s.gpu_coords, pop%(id_pre)s.gpu_r );
    """,

    #
    'init_connectivity_matrix': "",

    # Flattening operations
    'cuda_flattening': """
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
""",

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
    'sum': "local_res += local_r;",
    'mean': "local_res += local_r;"
}
cuda_pooling_code_2d = """
    int y_coords = centers[2*bIdx];
    int x_coords = centers[2*bIdx+1];
    int idx_x, idx_y;
    %(float_prec)s local_r = 0.0;
    %(float_prec)s local_res = %(sum_default)s;

    for( int y = 0; y < %(row_extent)s; y++ ) {
        idx_y = y_coords + y;
        for( int x = 0; x < %(col_extent)s; x++ ) {
            idx_x = x_coords + x;

            local_r = r [ idx_y * %(col_size)s + idx_x ];
%(operation)s
        }
    }
"""

cuda_pooling_code_3d = """
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
"""
