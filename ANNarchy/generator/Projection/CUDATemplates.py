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

%(cuda_flattening)s
};
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
"""
}

attribute_acc = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s > > get_%(name)s() { return %(name)s; }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return %(name)s[rk_post][rk_pre]; }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { %(name)s[rk] = value; %(name)s_dirty = true; }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { %(name)s[rk_post][rk_pre] = value; %(name)s_dirty = true; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector< std::vector<%(type)s> >(post_rank.size(), std::vector<%(type)s>());
        cudaMalloc((void**)&gpu_%(name)s, overallSynapses * sizeof(%(type)s));
        %(name)s_dirty = true;
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(post_rank.size(), %(init)s);
        cudaMalloc((void**)&gpu_%(name)s, post_rank.size() * sizeof(%(type)s));
        %(name)s_dirty = true;
"""
}

delay = {
    'declare': """
    // Non-uniform delay
    std::vector< std::vector< int > > delay ;""",
    'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s.delay = syn.delay""",
    'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
}

event_driven = {
    'declare': """
    std::vector<std::vector<long> > _last_event;
    long* gpu_last_event;
    void update_gpu_last_event() {
        std::vector<long> tmp =flattenArray<long>(_last_event);
        cudaMalloc((void**)&gpu_last_event, sizeof(long)*tmp.size());
        cudaMemcpy(gpu_last_event, tmp.data(), sizeof(long)*tmp.size(), cudaMemcpyHostToDevice);
        tmp.clear();
    }
""",
    'cpp_init': """
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
        void update_gpu_last_event()
""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id_proj)s._last_event[n] = vector[long](proj%(id_proj)s.nb_synapses(n), -10000)
        proj%(id_proj)s.update_gpu_last_event()
"""
}

######################################
### Connectivity matrix (pseudo CSR)
######################################
csr_connectivity_matrix = {
    'declare': """
    // Connectivity (LIL)
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

    // CSR
    int overallSynapses;
    std::vector<int> row_ptr;
    int *gpu_row_ptr;
    int* gpu_pre_rank;
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
        // row_ptr and overallSynapses
        genRowPtr();
    #ifdef _DEBUG_CONN
        std::cout << "Post to Pre:" << std::endl;
        for(int i = 0; i < pop%(id_post)s.size; i++) {
            std::cout << i << ": " << row_ptr[i] << " -> "<< row_ptr[i+1] << std::endl;
        }
    #endif
        cudaMalloc((void**)&gpu_row_ptr, row_ptr.size()*sizeof(int));
        cudaMemcpy(gpu_row_ptr, row_ptr.data(), row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice);

        // pre ranks
        std::vector<int> flat_pre_rank = flattenArray<int>(pre_rank);
        cudaMalloc((void**)&gpu_pre_rank, flat_pre_rank.size()*sizeof(int));
        cudaMemcpy(gpu_pre_rank, flat_pre_rank.data(), flat_pre_rank.size()*sizeof(int), cudaMemcpyHostToDevice);
""",
    'pyx_struct': """
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        # void inverse_connectivity_matrix()
""",
    'pyx_wrapper_args': " synapses",
    'pyx_wrapper_init': """
        cdef CSR syn = synapses
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
        # proj%(id_proj)s.inverse_connectivity_matrix() # TODO: spike only
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_rank()
    def set_pre_rank(self, val):
        proj%(id_proj)s.set_pre_rank(val)
        # proj%(id_proj)s.inverse_connectivity_matrix() ' TODO: spike only'
""",
}

csr_weight_matrix = {
    'declare': """
    // Local variable w
    std::vector<std::vector<double> > w;
    double *gpu_w;
    bool w_dirty;
    """,
    'accessor': """
    // Local variable w
    std::vector<std::vector< double > > get_w() { return w; }
    std::vector<double> get_dendrite_w(int rk) { return w[rk]; }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) { w = value; w_dirty = true; }
    void set_dendrite_w(int rk, std::vector<double> value) { w[rk] = value; w_dirty = true; }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; w_dirty = true; }
    """,
    'init': """
        // weights
        cudaMalloc((void**)&gpu_w, overallSynapses * sizeof(double));
        w_dirty = true; // enforce update
        cudaError_t err_w = cudaGetLastError();
        if ( err_w != cudaSuccess )
            std::cout << cudaGetErrorString(err_w) << std::endl;
""",
    'pyx_struct': """
        vector[ vector[ double] ] get_w()
        vector[ double ] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[ vector[ double] ])
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
"""

######################################
### Summation CUDA
######################################
# Comment to if (tid < 32) block:
#
# now that we are using warp-synchronous programming (below)
# we need to declare our shared memory volatile so that the compiler
# doesn't reorder stores to it and induce incorrect behavior.
cuda_psp_kernel = {
    'body': """
__global__ void cu_proj%(id_proj)s_psp( %(conn_args)s%(add_args)s, double* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int j = tid+row_ptr[blockIdx.x];

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(j < row_ptr[blockIdx.x+1])
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

    if (tid < 32)
    {
        volatile double* smem = sdata;

        if (blockDim.x >=  64) { smem[tid] = localSum = localSum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = localSum = localSum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = localSum = localSum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = localSum = localSum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = localSum = localSum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = localSum = localSum + smem[tid +  1]; }

    }

    // write result for this block to global mem
    if (tid == 0)
    {
        %(target_arg)s[blockIdx.x] = sdata[0];
    }

}
""",
    'header': """__global__ void cu_proj%(id)s_psp( %(conn_args)s%(add_args)s, double* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active ) {
        int sharedMemSize = __pop%(id_pre)s_pop%(id_post)s_%(target)s__ * 64;

        cu_proj%(id_proj)s_psp<<< pop%(id_post)s.size, __pop%(id_pre)s_pop%(id_post)s_%(target)s__, sharedMemSize>>>(
                       /* ranks and offsets */
                       %(conn_args)s
                       /* computation data */
                       %(add_args)s
                       /* result */
                       %(target_arg)s );
    }
"""

}

cuda_spike_psp_kernel=\
"""// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( double dt, int *spiked, %(conn_arg)s %(kernel_args)s ) {

    int pre_idx = spiked[blockIdx.x];
    int syn_idx = col_ptr[pre_idx]+threadIdx.x;

    //if (threadIdx.x == 0)
    //    printf("%%li - %%i: %%i, %%i\\n", t, pre_idx, col_ptr[pre_idx], col_ptr[pre_idx+1]);
    while( syn_idx < col_ptr[pre_idx+1]) {
%(event_driven)s
%(psp)s
%(pre_event)s

        syn_idx += blockDim.x;
    }
}
"""

cuda_spike_psp_kernel_call=\
"""
    if ( pop%(id_pre)s._active) {
        int num_events = pop%(id_pre)s.num_events;
        int tpb = __pop%(id_pre)s_pop%(id_post)s_%(target)s__;

    #ifdef _DEBUG
        std::cout << t << ": " << num_events << " event(s)." << std::endl;
    #endif
        if ( num_events > 0 ) {
            cu_proj%(id_proj)s_psp<<< num_events, tpb >>>( dt, pop%(id_pre)s.gpu_spiked, %(conn_args)s %(kernel_args)s );

        #ifdef _DEBUG
            cudaDeviceSynchronize();
            cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
            if( err_psp_proj%(id_proj)s != cudaSuccess) {
                std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
                std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
                std::cout << "   kernel_config: " << num_events << ", " << tpb << std::endl;
            }
        #endif
        }
    }
"""

######################################
### Update synaptic variables CUDA
######################################
cuda_synapse_kernel=\
"""
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_step( /* default params */
                              int *pre_rank, int* row_ptr, double dt
                              /* additional params */
                              %(var)s%(par)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int rk_post = blockIdx.x;
    int j = row_ptr[rk_post] + threadIdx.x;
    int C = row_ptr[rk_post+1];

    // Updating global variables of projection %(id)s
    if ( threadIdx.x == 0)
    {
%(global_eqs)s
    }

    // Updating local variables of projection %(id)s
    while ( j < C )
    {
        int rk_pre = pre_rank[j];

%(local_eqs)s

        j += blockDim.x;
    }
}
"""

cuda_synapse_kernel_header=\
"""__global__ void cuProj%(id)s_step( int *pre_rank, int *row_ptr, double dt%(var)s%(par)s, bool plasticity);
"""

cuda_synapse_kernel_call =\
"""
    // proj%(id_proj)s: pop%(pre)s -> pop%(post)s
    if ( proj%(id_proj)s._transmission && proj%(id_proj)s._update && proj%(id_proj)s._plasticity ) {
        cuProj%(id_proj)s_step<<< pop%(post)s.size, __pop%(pre)s_pop%(post)s_%(target)s__, 0, proj%(id_proj)s.stream>>>(
            proj%(id_proj)s.gpu_pre_rank,
            proj%(id_proj)s.gpu_row_ptr,
            dt
            %(local)s
            %(global)s
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t proj%(id_proj)s_step = cudaGetLastError();
        if (proj%(id_proj)s_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString(proj%(id_proj)s_step) << std::endl;
        }
    #endif
    }
"""

######################################
### post-event update CUDA
######################################
cuda_spike_postevent = {
    'body': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( double dt, int* spiked, int* row_ptr, int* pre_ranks, double* w %(add_args)s ) {
    int i = spiked[blockIdx.x];                // post-synaptic
    int j = row_ptr[i]+threadIdx.x;    // pre-synaptic

    while ( j < row_ptr[i+1] ) {
%(event_driven)s
%(post_code)s

        j+= blockDim.x;
    }
}
""",
    'header': """__global__ void cuProj%(id_proj)s_postevent( double dt, int* spiked, int* row_ptr, int* pre_ranks, double* w %(add_args)s );
""",
    'call': """
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active) {
        if (pop%(id_post)s.num_events > 0 ) {
            cuProj%(id_proj)s_postevent<<< pop%(id_post)s.num_events, __pop%(id_pre)s_pop%(id_post)s_%(target)s__ >>>(
                dt, pop%(id_post)s.gpu_spiked
                /* connectivity */
                , proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_row_ptr
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

cuda_templates = {
    'projection_header': projection_header,
    'attribute_decl': attribute_decl,
    'attribute_acc':attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,

    'delay': delay,
    'event_driven': event_driven,

    'cuda_stream': cuda_stream
}