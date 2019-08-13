#===============================================================================
#
#     LIL_CUDA.py
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
"""
This file contains all template codes to represent the connectivity as a
compressed sparse row and column data structure.

Please remember, that the interface to the PyExtension remains as a
list of list data structure.

All templates are gathered in a dictionary called conn_templates,
which should be used in CUDAGenerator or CUDAConnectivity.
"""
connectivity_matrix = {
    'declare': """
    // Connectivity (LIL)
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

    // CSR
    int overallSynapses;
    std::vector<int> row_ptr;
    int *gpu_row_ptr;
    int* gpu_pre_rank;
    int *gpu_post_rank;
""",
    'accessor': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }
""",
    'init': """
        // compute row_ptr and overallSynapses
        genRowPtr();

    #ifdef _DEBUG_CONN
        std::cout << "pop%(id_pre)s->pop%(id_post)s ( %(target)s )" << std::endl;
        std::cout << "  Post to Pre:" << std::endl;
        for(int i = 0; i < pop%(id_post)s.size; i++) {
            std::cout << "    " << i << ": " << row_ptr[i] << " -> "<< row_ptr[i+1] << std::endl;
        }
        std::cout << "  contains " << overallSynapses << " synapses." << std::endl;
    #endif

        // transfer row pointer
        cudaMalloc((void**)&gpu_row_ptr, row_ptr.size()*sizeof(int));
        cudaMemcpy(gpu_row_ptr, row_ptr.data(), row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_row_ptr = cudaGetLastError();
        if ( err_row_ptr != cudaSuccess )
            std::cout << "HtoD: row_ptr (proj%(id_proj)s) " << cudaGetErrorString(err_row_ptr) << std::endl;
    #endif

        // transfer post ranks
        cudaMalloc((void**)&gpu_post_rank, post_rank.size() * sizeof(int));
        cudaMemcpy(gpu_post_rank, post_rank.data(), post_rank.size() * sizeof(int), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_post_rank = cudaGetLastError();
        if ( err_post_rank != cudaSuccess )
            std::cout << "HtoD: post_rank (proj%(id_proj)s) " << cudaGetErrorString(err_post_rank) << std::endl;
    #endif

        // transfer pre ranks
        std::vector<int> flat_pre_rank = flattenArray<int>(pre_rank);
        cudaMalloc((void**)&gpu_pre_rank, flat_pre_rank.size()*sizeof(int));
        cudaMemcpy(gpu_pre_rank, flat_pre_rank.data(), flat_pre_rank.size()*sizeof(int), cudaMemcpyHostToDevice);
    #ifdef _DEBUG
        cudaError_t err_pre_rank = cudaGetLastError();
        if ( err_pre_rank != cudaSuccess )
            std::cout << "HtoD: pre_rank (proj%(id_proj)s) " << cudaGetErrorString(err_pre_rank) << std::endl;
    #endif
""",
    'clear': """
    cudaFree(gpu_row_ptr);
    cudaFree(gpu_post_rank);
    cudaFree(gpu_pre_rank);
#ifdef _DEBUG
    cudaError_t err_clear_conn = cudaGetLastError();
    if ( err_clear_conn != cudaSuccess )
        std::cout << "Proj%(id_proj)::clear() - connectivity: " << cudaGetErrorString(err_clear_conn) << std::endl;
#endif
""",
    'pyx_struct': """
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
""",
    'pyx_wrapper_args': " synapses",
    'pyx_wrapper_init': """
        cdef LIL syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj%(id_proj)s.set_size( size )
        proj%(id_proj)s.set_post_rank( syn.post_rank )
        proj%(id_proj)s.set_pre_rank( syn.pre_rank )
""",
    'pyx_wrapper_accessor': """
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def set_post_rank(self, val):
        proj%(id_proj)s.set_post_rank(val)
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_rank()
    def set_pre_rank(self, val):
        proj%(id_proj)s.set_pre_rank(val)
""",
}

weight_matrix = {
    'declare': """
    // Local variable w
    std::vector<std::vector< %(float_prec)s > > w;
    %(float_prec)s *gpu_w;
    bool w_dirty;
    """,
    'accessor': """
    // Local variable w
    std::vector<std::vector<double> > get_w()  { 
        std::vector< std::vector< double > > w_new(w.size(), std::vector<double>());
        for(int i = 0; i < w.size(); i++) {
            w_new[i] = std::vector<double>(w[i].begin(), w[i].end());
        }
        return w_new;
    }
    std::vector<double> get_dendrite_w(int rk) {
        return std::vector<double>(w[rk].begin(), w[rk].end());
    }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector<double> >value) {
        w = std::vector< std::vector<%(float_prec)s> >( value.size(), std::vector<%(float_prec)s>() );
        for(int i = 0; i < value.size(); i++) {
            w[i] = std::vector<%(float_prec)s>(value[i].begin(), value[i].end());
        }
        w_dirty = true;
    }
    void set_dendrite_w(int rk, std::vector<double> value) {
        w[rk] = std::vector<%(float_prec)s>(value.begin(), value.end());
        w_dirty = true; 
    }
    void set_synapse_w(int rk_post, int rk_pre, %(float_prec)s value) { w[rk_post][rk_pre] = value; w_dirty = true; }
    """,
    'init': """
        // weights
        cudaMalloc((void**)&gpu_w, overallSynapses * sizeof(%(float_prec)s));
        w_dirty = true; // enforce update
        cudaError_t err_w = cudaGetLastError();
        if ( err_w != cudaSuccess )
            std::cout << cudaGetErrorString(err_w) << std::endl;
""",
    'clear': """
    cudaFree(gpu_w);
""",
    'pyx_struct': """
        vector[ vector[ double ] ] get_w()
        vector[ double ] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[ vector[ double ] ])
        void set_dendrite_w( int, vector[double])
        void set_synapse_w(int, int, double)
    """,
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w(syn.w)
    """,
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def set_w(self, value):
        proj%(id_proj)s.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj%(id_proj)s.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.set_synapse_w(rank_post, rank_pre, value)
"""
}

single_weight_matrix = {
    'declare': """
    // Single weight in the projection
    // TODO:
""",
    'accessor': "",
    'init': "",
    'pyx_struct': """
        # Local variable w
        # TODO:
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        # Use only the first weight
        # TODO:
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    # TODO:
"""
}

inverse_connectivity_matrix = {
    'declare': """
    // Inverse connectivity, only on gpu
    int* gpu_col_ptr;
    int* gpu_row_idx;
    int* gpu_inv_idx;
""",
    'init': """
        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        std::vector< std::vector< int > > pre_to_post_rank = std::vector< std::vector< int > >(pop%(id_pre)s.size, std::vector<int>());
        std::vector< std::vector< int > > pre_to_post_idx = std::vector< std::vector< int > >(pop%(id_pre)s.size, std::vector<int>());

        // some iterator definitions we need
        typename std::vector<std::vector<int> >::iterator pre_rank_out_it = pre_rank.begin();  // 1st level iterator
        typename std::vector<int>::iterator pre_rank_in_it;                                    // 2nd level iterator
        typename std::vector< int >::iterator post_rank_it = post_rank.begin();

        // iterate over post neurons, post_rank_it encodes the current rank
        for( ; pre_rank_out_it != pre_rank.end(); pre_rank_out_it++, post_rank_it++ ) {

            int syn_idx = row_ptr[*post_rank_it]; // start point of the flattened array, post-side
            // iterate over synapses, update both result containers
            for( pre_rank_in_it = pre_rank_out_it->begin(); pre_rank_in_it != pre_rank_out_it->end(); pre_rank_in_it++) {
                //std::cout << *pre_rank_in_it << "->" << *post_rank_it << ": " << syn_idx << std::endl;
                pre_to_post_rank[*pre_rank_in_it].push_back(*post_rank_it);
                pre_to_post_idx[*pre_rank_in_it].push_back(syn_idx);
                syn_idx++;
            }
        }

        std::vector<int> col_ptr = std::vector<int>( pop%(id_pre)s.size, 0 );
        int curr_off = 0;
        for ( int i = 0; i < pop%(id_pre)s.size; i++) {
            col_ptr[i] = curr_off;
            curr_off += pre_to_post_rank[i].size();
        }
        col_ptr.push_back(curr_off);

    #ifdef _DEBUG_CONN
        std::cout << "Pre to Post:" << std::endl;
        for ( int i = 0; i < pop%(id_pre)s.size; i++ ) {
            std::cout << i << ": " << col_ptr[i] << " -> " << col_ptr[i+1] << std::endl;
        }
    #endif

        cudaMalloc((void**)&gpu_col_ptr, col_ptr.size()*sizeof(int));
        cudaMemcpy(gpu_col_ptr, col_ptr.data(), col_ptr.size()*sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> row_idx = flattenArray(pre_to_post_rank);
        cudaMalloc((void**)&gpu_row_idx, row_idx.size()*sizeof(int));
        cudaMemcpy(gpu_row_idx, row_idx.data(), row_idx.size()*sizeof(int), cudaMemcpyHostToDevice);

        std::vector<int> inv_idx = flattenArray(pre_to_post_idx);
        cudaMalloc((void**)&gpu_inv_idx, inv_idx.size()*sizeof(int));
        cudaMemcpy(gpu_inv_idx, inv_idx.data(), inv_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
"""
}

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
"""
}

attribute_acc = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s > > get_%(name)s() { return %(name)s; }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return %(name)s[rk_post][rk_pre]; }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { %(name)s[rk] = value; %(name)s_dirty = true; }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { %(name)s[rk_post][rk_pre] = value; %(name)s_dirty = true; }
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; %(name)s_dirty = true; }
""",
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s( %(type)s value ) { %(name)s = value; %(name)s_dirty = true; }
"""
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector< std::vector<%(type)s> >(post_rank.size(), std::vector<%(type)s>());
        cudaMalloc((void**)&gpu_%(name)s, overallSynapses * sizeof(%(type)s));
        %(name)s_dirty = true;
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(post_rank.size(), %(init)s);
        cudaMalloc((void**)&gpu_%(name)s, post_rank.size() * sizeof(%(type)s));
        %(name)s_dirty = true;
""",
    'global': """
        // Global %(attr_type)s %(name)s
        %(name)s = %(type)s(0);
        cudaMalloc((void**)&gpu_%(name)s, sizeof(%(type)s));
        %(name)s_dirty = true;
"""
}

attribute_host_to_device = {
    'local': """
        // %(name)s: local
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s, synaptic variable )" << std::endl;
        #endif
            std::vector< %(type)s > flat_%(name)s = flattenArray< %(type)s >( %(name)s );
            cudaMemcpy( gpu_%(name)s, flat_%(name)s.data(), flat_%(name)s.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'semiglobal': """
        // %(name)s: semiglobal
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s, post-synaptic variable )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), post_rank.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'global': """
        // %(name)s: global
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s, projection variable )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, &%(name)s, sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
"""
}

attribute_device_to_host = {
    'local': """
            // %(name)s: local
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            std::vector<%(type)s> flat_%(name)s = std::vector<%(type)s>( overallSynapses, 0);
            cudaMemcpy(flat_%(name)s.data(), gpu_%(name)s, flat_%(name)s.size() * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
            %(name)s = deFlattenArray< %(type)s >( flat_%(name)s );
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
""",
    'semiglobal': """
            // %(name)s: semiglobal
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, post_rank.size() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
""",
    'global': """
            // %(name)s: global
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( &%(name)s, gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
"""
}

delay = {
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;""",
        'pyx_struct': """
        # Non-uniform delay
        int delay""",
        'init': "",
        'pyx_wrapper_init': """
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor': """
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""},
    'nonuniform': {
        'declare': """
    // Non-uniform delay
    std::vector< std::vector< int > > delay ;""",
        'pyx_struct': """
        # Non-uniform delay
        vector[vector[int]] delay""",
        'init': "",
        'pyx_wrapper_init': """
        proj%(id_proj)s.delay = syn.delay""",
        'pyx_wrapper_accessor': """
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
    }
}

event_driven = {
    'declare': """
    std::vector<std::vector<long> > _last_event;
    long* _gpu_last_event;
    void update_gpu_last_event() {
        std::vector<long> tmp =flattenArray<long>(_last_event);
        cudaMalloc((void**)&_gpu_last_event, sizeof(long)*tmp.size());
        cudaMemcpy(_gpu_last_event, tmp.data(), sizeof(long)*tmp.size(), cudaMemcpyHostToDevice);
        tmp.clear();
    }
""",
    'cpp_init': """
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
        void update_gpu_last_event()
""",
    'pyx_wrapper_init': """
        proj%(id_proj)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id_proj)s._last_event[n] = vector[long](proj%(id_proj)s.nb_synapses(n), -10000)
        proj%(id_proj)s.update_gpu_last_event()
"""
}

#
# Implement the weighte sum operation for rate-code synapses.
#
rate_psp_kernel = {
    # Comment to if (tid < 32) block:
    #
    # now that we are using warp-synchronous programming (below)
    # we need to declare our shared memory volatile so that the compiler
    # doesn't reorder stores to it and induce incorrect behavior.
    'body': {
        'sum':"""
__global__ void cu_proj%(id_proj)s_psp( int post_size, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        unsigned int j = tid+row_ptr[bid];

        %(float_prec)s localSum = %(thread_init)s;
        while(j < row_ptr[bid+1])
        {
            localSum += %(psp)s

            j+= blockDim.x;
        }

        sdata[tid] = localSum;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] = localSum = localSum + sdata[tid +  32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            smem[tid] = localSum = localSum + smem[tid + 16];
            smem[tid] = localSum = localSum + smem[tid +  8];
            smem[tid] = localSum = localSum + smem[tid +  4];
            smem[tid] = localSum = localSum + smem[tid +  2];
            smem[tid] = localSum = localSum + smem[tid +  1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[bid] += sdata[0];
        }

        bid += gridDim.x;
    }
}
""",
    'min':"""
__global__ void cu_proj%(id_proj)s_psp( int post_size, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        unsigned int j = tid+row_ptr[bid];

        // Init all threads with max. value
        %(float_prec)s localMin = %(thread_init)s;

        // Iterate with chunks over the array
        while(j < row_ptr[bid+1])
        {
            auto tmp = %(psp)s;
            if (tmp < localMin)
                localMin = tmp;

            j+= blockDim.x;
        }

        sdata[tid] = localMin;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { if ( sdata[tid] > sdata[tid + 256] ) sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { if ( sdata[tid] > sdata[tid + 128] ) sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { if ( sdata[tid] > sdata[tid + 64] ) sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { if ( sdata[tid] > sdata[tid + 32] ) sdata[tid] = sdata[tid + 32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            // if other value is smaller, copy
            if ( smem[tid] > smem[tid + 16] ) smem[tid] = smem[tid + 16];
            if ( smem[tid] > smem[tid +  8] ) smem[tid] = smem[tid + 8];
            if ( smem[tid] > smem[tid +  4] ) smem[tid] = smem[tid + 4];
            if ( smem[tid] > smem[tid +  2] ) smem[tid] = smem[tid + 2];
            if ( smem[tid] > smem[tid +  1] ) smem[tid] = smem[tid + 1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[bid] += sdata[0];
        }

        bid += gridDim.x;
    }
}
""",
    'max':"""
__global__ void cu_proj%(id_proj)s_psp( int post_size, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        unsigned int j = tid+row_ptr[bid];

        // Init all threads with min. value
        %(float_prec)s localMax = %(thread_init)s;

        // Iterate with chunks over the array
        while(j < row_ptr[bid+1])
        {
            %(float_prec)s tmp = %(psp)s;
            if (tmp > localMax)
                localMax = tmp;

            j+= blockDim.x;
        }

        sdata[tid] = localMax;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { if ( sdata[tid] < sdata[tid + 256] ) sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { if ( sdata[tid] < sdata[tid + 128] ) sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { if ( sdata[tid] < sdata[tid + 64] ) sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { if ( sdata[tid] < sdata[tid + 32] ) sdata[tid] = sdata[tid + 32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            // if other value is larger, copy
            if ( smem[tid] < smem[tid + 16] ) smem[tid] = smem[tid + 16];
            if ( smem[tid] < smem[tid +  8] ) smem[tid] = smem[tid + 8];
            if ( smem[tid] < smem[tid +  4] ) smem[tid] = smem[tid + 4];
            if ( smem[tid] < smem[tid +  2] ) smem[tid] = smem[tid + 2];
            if ( smem[tid] < smem[tid +  1] ) smem[tid] = smem[tid + 1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[bid] += sdata[0];
        }

        bid += gridDim.x;
    }
}
""",
    # Technically a sum operation, but the result is normalized with the number of connection entries
    'mean': """
__global__ void cu_proj%(id_proj)s_psp( int post_size, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        unsigned int j = tid+row_ptr[bid];

        %(float_prec)s localSum = %(thread_init)s;
        while(j < row_ptr[bid+1])
        {
            localSum += %(psp)s

            j+= blockDim.x;
        }

        sdata[tid] = localSum;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] = localSum = localSum + sdata[tid +  32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            smem[tid] = localSum = localSum + smem[tid + 16];
            smem[tid] = localSum = localSum + smem[tid +  8];
            smem[tid] = localSum = localSum + smem[tid +  4];
            smem[tid] = localSum = localSum + smem[tid +  2];
            smem[tid] = localSum = localSum + smem[tid +  1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[bid] += sdata[0] / (%(float_prec)s(row_ptr[bid+1]-row_ptr[bid]));
        }

        bid += gridDim.x;
    }
}
"""
    },
    'header': """__global__ void cu_proj%(id)s_psp( int post_size, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        int sharedMemSize = __proj%(id_proj)s_%(target)s_tpb__ * sizeof(%(float_prec)s);

        cu_proj%(id_proj)s_psp<<< __proj%(id_proj)s_%(target)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, sharedMemSize>>>(
                       pop%(id_post)s.size,
                       /* ranks and offsets */
                       %(conn_args)s
                       /* computation data */
                       %(add_args)s
                       /* result */
                       %(target_arg)s );

    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
""",
    'thread_init': {
        'float': {
            'sum': "0.0f",
            'min': "FLT_MAX",
            'max': "FLT_MIN",
            'mean': "0.0f"
        },
        'double': {
            'sum': "0.0",
            'min': "DBL_MAX",
            'max': "DBL_MIN",
            'mean': "0.0"
        }
    },

    # EXPERIMENTAL
    'one2one': """
// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( %(float_prec)s dt, bool plasticity, int *spiked, %(conn_arg)s %(kernel_args)s ) {
    int syn_idx = spiked[blockIdx.x]; // one2one: syn_idx = n_idx

    if(threadIdx.x == 0) {
        g_target[syn_idx] += w[syn_idx];
        if ( g_target[syn_idx] > max_trans[syn_idx] )
            g_target[syn_idx] = max_trans[syn_idx];
    }
}
"""
}

spike_event_transmission = {
    #
    # This kernel computes the post-synaptic potential for post1st structures and
    # uses the inverse connectivty data for this purpose.
    'post_to_pre': {
        'body': """// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_arg)s %(kernel_args)s ) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    while ( bid < *num_events ) {
        int pre_index = spiked[bid];

        int j = col_ptr[pre_index] + tid;

        while ( j < col_ptr[pre_index+1] ) {
            int syn_idx = inv_idx[j];
            int post_rank = row_idx[j];

            // event-driven
%(event_driven)s

            // increase of conductance
%(psp)s

            // pre-spike statements
%(pre_event)s

            j += blockDim.x;
        }

        bid += gridDim.x;
    }
}
""",
        'header': """__global__ void cu_proj%(id)s_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_header)s %(kernel_args)s);
""",
        'call': """
    if ( pop%(id_pre)s._active && (pop%(id_pre)s.spike_count > 0) && proj%(id_proj)s._transmission ) {
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;

        // compute psp using backward view ...
        cu_proj%(id_proj)s_psp<<< int(pop%(id_pre)s.spike_count), tpb, 0, proj%(id_proj)s.stream >>>( 
            dt, proj%(id_proj)s._plasticity, pop%(id_pre)s.gpu_spiked, pop%(id_pre)s.gpu_spike_count, 
            /* connectivity */
            %(conn_args)s
            /* kernel config */
            %(kernel_args)s
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
        if( err_psp_proj%(id_proj)s != cudaSuccess) {
            std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
            std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
        }
    #endif
    }
"""
    }
}

spike_continous_transmission = {
    #
    # This kernel computes the post-synaptic potential for continous
    # transmission using the forward view of connectivty data.
    #
    # ATTENTION: post_idx and post_rank diverge in case of non-existant
    #            dendrites
    #
    # TODO: it might be more effective to split this kernel into two functions ...
    'post_to_pre': {
        'body': """// gpu device kernel for projection %(id_proj)s
__global__ void cu_proj%(id_proj)s_event_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, 
                                              /* connectivity */
                                              int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s *w
                                              /* additional arguments */
                                              %(kernel_args)s 
                                            ) 
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    while ( bid < *num_events ) {
        int pre_index = spiked[bid];

        int j = col_ptr[pre_index] + tid;

        // pre-spike statements
        while ( j < col_ptr[pre_index+1] ) {
            int syn_idx = inv_idx[j];
            int post_rank = row_idx[j];

%(pre_code)s

            j += blockDim.x;
        }

        bid += gridDim.x;
    }
}

__global__ void cu_proj%(id_proj)s_cont_psp( %(float_prec)s dt, bool plasticity, int post_size, int* post_ranks, 
                                            /* connectivity */
                                            int* row_ptr, int *col_idx, %(float_prec)s *w
                                            /* additional arguments */
                                            %(kernel_args)s
                                            /* target */
                                            , %(float_prec)s* %(target_arg)s ) 
{
    int post_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern %(float_prec)s __shared__ sdata[];

    while ( post_idx < post_size ) {
        // which dendrite we are working on
        int post_rank = post_ranks[post_idx];
        int syn_idx = row_ptr[post_rank] + tid;

        %(float_prec)s localSum = 0.0;

        while( syn_idx < row_ptr[post_rank+1] ) {
            localSum += %(psp)s
            syn_idx += blockDim.x;
        }

        sdata[tid] = localSum;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] = localSum = localSum + sdata[tid +  32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            smem[tid] = localSum = localSum + smem[tid + 16];
            smem[tid] = localSum = localSum + smem[tid +  8];
            smem[tid] = localSum = localSum + smem[tid +  4];
            smem[tid] = localSum = localSum + smem[tid +  2];
            smem[tid] = localSum = localSum + smem[tid +  1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[post_rank] += sdata[0];
        }
        __syncthreads();
        post_idx += gridDim.x;
    }
}
""",
        'header': """__global__ void cu_proj%(id)s_event_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s *w %(kernel_args)s);
__global__ void cu_proj%(id)s_cont_psp( %(float_prec)s dt, bool plasticity, int post_size, int* post_ranks, int* row_ptr, int *col_idx, %(float_prec)s *w %(kernel_args)s, %(float_prec)s* %(target_arg)s );
""",
        'call': """
    if ( pop%(id_pre)s._active && proj%(id_proj)s._transmission ) {
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;

        if (pop%(id_pre)s.spike_count > 0) {
            // compute event-based transmission using backward view ...
            cu_proj%(id_proj)s_event_psp<<< int(pop%(id_pre)s.spike_count), tpb, 0, proj%(id_proj)s.stream >>>( 
                dt, proj%(id_proj)s._plasticity, pop%(id_pre)s.gpu_spiked, pop%(id_pre)s.gpu_spike_count, 
                /* connectivity */
                proj%(id_proj)s.gpu_col_ptr, proj%(id_proj)s.gpu_row_idx, proj%(id_proj)s.gpu_inv_idx, proj%(id_proj)s.gpu_w
                /* kernel config */
                %(kernel_args)s
            );
        }

        // compute continous transmission using forward view ...
        cu_proj%(id_proj)s_cont_psp<<< proj%(id_proj)s.post_rank.size(), tpb, tpb*sizeof(%(float_prec)s), proj%(id_proj)s.stream >>>( 
            dt, proj%(id_proj)s._plasticity, proj%(id_proj)s.post_rank.size(), proj%(id_proj)s.gpu_post_rank, 
            /* connectivity */
            proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_w
            /* additional arguments */ 
            %(kernel_args)s
            /* target */
            %(target_arg)s );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
        if( err_psp_proj%(id_proj)s != cudaSuccess) {
            std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
            std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
        }
    #endif
    }
"""
    }
}

synapse_update = {
    # Update of global synaptic equations, consist of body (annarchyDevice.cu),
    # header and call semantic (take place in ANNarchyHost.cu)
    'global': {
        'body': """
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_global_step( /* default params */
                              int post_size, int *pre_rank, int *row_ptr, %(float_prec)s dt
                              /* additional params */
                              %(kernel_args)s,
                              /* plasticity enabled */
                              bool plasticity )
{
%(pre_loop)s
%(global_eqs)s
}
""",
        'header': """__global__ void cuProj%(id)s_global_step( int post_size, int *pre_rank, int *row_ptr, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // global update
        cuProj%(id_proj)s_global_step<<< 1, 1, 0, proj%(id_proj)s.stream>>>(
            proj%(id_proj)s.post_rank.size(),
            /* default args*/
            proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( global_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
"""
    },

    # Update of semiglobal synaptic equations, consist of body (annarchyDevice.cu),
    # header and call semantic (take place in ANNarchyHost.cu)
    'semiglobal': {
        'body': """
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_semiglobal_step( /* default params */
                              int post_size, int *pre_rank, int *row_ptr, %(float_prec)s dt
                              /* additional params */
                              %(kernel_args)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int rk_post = threadIdx.x + blockIdx.x*blockDim.x;
%(pre_loop)s
    while ( rk_post < post_size ) {
%(semiglobal_eqs)s

        rk_post += gridDim.x * blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id)s_semiglobal_step( int post_size, int *pre_rank, int *row_ptr, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // semiglobal update
        nb_blocks = ceil( %(float_prec)s(proj%(id_proj)s.post_rank.size()) / %(float_prec)s(__proj%(id_proj)s_%(target)s_tpb__));
        cuProj%(id_proj)s_semiglobal_step<<< nb_blocks, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream >>>(
            proj%(id_proj)s.post_rank.size(),
            /* default args*/
            proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess) {
            std::cout << "proj%(id_proj)s_semiglobal_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
"""
    },

    # Update of local synaptic equations, consist of body (annarchyDevice.cu),
    # header and call semantic (take place in ANNarchyHost.cu)
    'local': {
        'body': """
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_local_step( /* default params */
                              int *post_rank, int *pre_rank, int *row_ptr, %(float_prec)s dt
                              /* additional params */
                              %(kernel_args)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int rk_post = post_rank[blockIdx.x];
    int j = row_ptr[rk_post] + threadIdx.x;
    int C = row_ptr[rk_post+1];
%(pre_loop)s

    // Updating local variables of projection %(id)s
    while ( j < C )
    {
%(local_eqs)s

        j += blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id)s_local_step( int *post_rank, int *pre_rank, int *row_ptr, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // local update
        cuProj%(id_proj)s_local_step<<< pop%(id_post)s.size, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream >>>(
            /* default args*/
            proj%(id_proj)s.gpu_post_rank, proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
""",
    },

    # call semantic for global, semiglobal and local kernel
    'call': """
    // proj%(id_proj)s: pop%(pre)s -> pop%(post)s
    if ( proj%(id_proj)s._transmission && proj%(id_proj)s._update && proj%(id_proj)s._plasticity && ( (t - proj%(id_proj)s._update_offset)%%proj%(id_proj)s._update_period == 0L)) {
        %(float_prec)s _dt = dt * proj%(id_proj)s._update_period;
#ifdef _DEBUG
    cudaError_t err;
#endif
%(global_call)s
int nb_blocks;
%(semiglobal_call)s
%(local_call)s
    }
"""
}

#
# Evaluation of post-event equations
#
spike_postevent = {
    'post_to_pre': {
        #
        # Called if storage_order is 'post_to_pre'. The vector pop%(id).gpu_spiked must be interpreted
        # as a boolean array. The parallelization happens across pop%(id).spike_count blocks.
        #
        'body': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s ) {
    int i = post_rank[spiked[blockIdx.x]]; // post-synaptic
    int j = row_ptr[i]+threadIdx.x;        // pre-synaptic

    while ( j < row_ptr[i+1] ) {
    // event-driven
%(event_driven)s

    // post-event
%(post_code)s

        j+= blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id_proj)s_postevent( %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s );
""",
        # Each cuda block compute one of the spiking post-synaptic neurons
        'call': """
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active && (pop%(id_post)s.spike_count > 0) ) {

        cuProj%(id_proj)s_postevent<<< pop%(id_post)s.spike_count, __proj%(id_proj)s_%(target)s_tpb__ >>>(
            dt, proj%(id_proj)s._plasticity, proj%(id_proj)s.gpu_post_rank, pop%(id_post)s.gpu_spiked
            /* connectivity */
            %(conn_args)s
            /* weights */
            , proj%(id_proj)s.gpu_w
            /* other variables */
            %(add_args)s
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t proj%(id_proj)s_postevent = cudaGetLastError();
        if (proj%(id_proj)s_postevent != cudaSuccess) {
            std::cout << "proj%(id_proj)s_postevent: " << cudaGetErrorString(proj%(id_proj)s_postevent) << std::endl;
        }
    #endif
    }
"""
    },
    'pre_to_post': {
        #
        # pop%(id).gpu_spiked contains pop%(id).spike_count indices.
        # The parallelization happens across pop%(id).spike_count blocks.
        #
        # TODO: validate correctness.
        'body': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( %(float_prec)s dt, bool plasticity, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s ) {
    int i = spiked[blockIdx.x];                // post-synaptic
    int j = row_ptr[i]+threadIdx.x;    // pre-synaptic

    while ( j < row_ptr[i+1] ) {
%(event_driven)s
%(post_code)s

        j+= blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id_proj)s_postevent( %(float_prec)s dt, bool plasticity, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s );
""",
        'call': """
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active) {
        if (pop%(id_post)s.spike_count > 0 ) {
            cuProj%(id_proj)s_postevent<<< pop%(id_post)s.spike_count, __pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__ >>>(
                dt, proj%(id_proj)s._plasticity, pop%(id_post)s.gpu_spiked
                /* connectivity */
                %(conn_args)s
                /* weights */
                , proj%(id_proj)s.gpu_w
                /* other variables */
                %(add_args)s
            );
        #ifdef _DEBUG
            cudaDeviceSynchronize();
            cudaError_t proj%(id_proj)s_postevent = cudaGetLastError();
            if (proj%(id_proj)s_postevent != cudaSuccess) {
                std::cout << "proj%(id_proj)s_postevent: " << cudaGetErrorString(proj%(id_proj)s_postevent) << std::endl;
            }
        #endif
        }
    }
"""
    }
}

# Summation of all fields
conn_templates = {
    # connectivity
    'connectivity_matrix': connectivity_matrix,
    'inverse_connectivity_matrix': inverse_connectivity_matrix,
    'weight_matrix': weight_matrix,
    'single_weight_matrix': single_weight_matrix,

    # accessors
    'attribute_decl': attribute_decl,
    'attribute_acc':attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,
    'delay': delay,
    'event_driven': event_driven,

    # operations
    'rate_psp': rate_psp_kernel,
    'spike_transmission': {
        'event_driven': spike_event_transmission,
        'continous': spike_continous_transmission,
    },
    'synapse_update': synapse_update,
    'post_event': spike_postevent,
}
