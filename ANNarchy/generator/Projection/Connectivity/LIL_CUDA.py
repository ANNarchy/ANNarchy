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
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
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
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; %(name)s_dirty = true; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s( %(type)s value ) { %(name)s = value; %(name)s_dirty = true; }
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
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(post_rank.size(), %(init)s);
        cudaMalloc((void**)&gpu_%(name)s, post_rank.size() * sizeof(%(type)s));
        %(name)s_dirty = true;
""",
    'global':
"""
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
            std::vector<double> flat_%(name)s = flattenArray<double>( %(name)s );
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
    'pyx_wrapper_init':
"""
        proj%(id_proj)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id_proj)s._last_event[n] = vector[long](proj%(id_proj)s.nb_synapses(n), -10000)
        proj%(id_proj)s.update_gpu_last_event()
"""
}

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
    'event_driven': event_driven
}