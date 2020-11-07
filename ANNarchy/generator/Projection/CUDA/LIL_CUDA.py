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
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) {
        update_matrix_variable_all(%(name)s, value);
        %(name)s_dirty = true; 
    }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) {
        update_matrix_variable_row(%(name)s, rk, value);
        %(name)s_dirty = true; 
    }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) {
        update_matrix_variable(%(name)s, rk_post, rk_pre, value);
        %(name)s_dirty = true; 
    }
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
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_dirty = true;
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_vector_variable_gpu<%(type)s>(%(name)s);
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
            std::vector<%(type)s> flat_%(name)s = std::vector<%(type)s>( num_non_zeros_, 0);
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
        'init': """
        delay = delays[0][0];
""",
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
    'nonuniform_rate_coded': None,
    'nonuniform_spiking': None
}

event_driven = {
    'declare': """
    std::vector<std::vector<long> > _last_event;
    long* _gpu_last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long>(-10000);
    _gpu_last_event = init_matrix_variable_gpu<long>(_last_event);
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
                       proj%(id_proj)s.nb_dendrites(),
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
