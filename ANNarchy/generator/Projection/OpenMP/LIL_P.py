"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector< std::vector<%(type)s > > > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable< %(type)s, std::vector<std::vector<%(type)s>> >(%(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(%(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
}

attribute_cpp_size = {
    'local': """
        // Local %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<std::vector<%(ctype)s>>);
        size_in_bytes += sizeof(std::vector<%(ctype)s>) * %(name)s.capacity();
        for(auto it = %(name)s.cbegin(); it != %(name)s.cend(); it++)
            size_in_bytes += (it->capacity()) * sizeof(%(ctype)s);
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global %(attr_type)s %(name)s
        size_in_bytes += sizeof(%(ctype)s);
"""
}

attribute_cpp_delete = {
    'local': """
        // %(name)s
        for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        };
        %(name)s.clear();
        %(name)s.shrink_to_fit();
""",
    'semiglobal': """
        // %(name)s
        %(name)s.clear();
        %(name)s.shrink_to_fit();
""",
    'global': ""
}

cpp_11_rng = {
    'template': """#pragma omp single
{
    %(global_rng)s
    for(std::vector<%(idx_type)s>::size_type i = 0; i < post_rank.size(); i++) {
    %(semiglobal_rng)s
        for(std::vector<%(idx_type)s>::size_type j = 0; j < pre_rank[i].size(); j++) {
    %(local_rng)s
        }
    }
}
""",
    'global': """
%(rd_name)s = dist_%(rd_name)s(rng[0]);
""",
    'semiglobal': """
    %(rd_name)s[i] = dist_%(rd_name)s(rng[0]);
""",
    'local': """
        %(rd_name)s[i][j] = dist_%(rd_name)s(rng[0]);
"""
}

delay = {
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;""",
        
        'pyx_struct':
"""
        # Uniform delay
        int delay""",
        'init': """
    delay = delays[0][0];
""",
        'pyx_wrapper_init':
"""
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""},
    'nonuniform_rate_coded': {
        'declare': """
    std::vector<std::vector<std::vector<int>>> delay;
    int max_delay;

    std::vector<int> get_dendrite_delay(int row) {
        return get_matrix_variable_row<int, std::vector<std::vector<int>>>(delay, row);
    }
    std::vector<std::vector<int>> get_delay() {
        return get_matrix_variable_all<int, std::vector<std::vector<int>>>(delay);
    }
    void set_dendrite_delay(int row, std::vector<int> value) {
        update_matrix_variable_row<int, std::vector<std::vector<int>>>(delay, row, value);
    }
    void set_delay(std::vector<std::vector<int>> value) {
        update_matrix_variable_all<int, std::vector<std::vector<int>>>(delay, value);
    }
""",
        'init': """
    delay = init_matrix_variable<int, std::vector<std::vector<int>>>(1);
    update_matrix_variable_all<int>(delay, delays);

    max_delay = %(pre_prefix)smax_delay;
""",
        'reset': "",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] get_delay()
        vector[int] get_dendrite_delay(int)
        void set_delay(vector[vector[int]])
        void set_dendrite_delay(int, vector[int])
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.get_delay()
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.get_dendrite_delay(idx)
    def set_delay(self, value):
        proj%(id_proj)s.set_delay(value)
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    },
    'nonuniform_spiking': {
        'declare': """
    // Nonuniform spiking delays
    std::vector<std::vector<std::vector<int>>> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< std::vector< int > > > > _delayed_spikes;

    std::vector<std::vector<int>> get_delay() {
        return get_matrix_variable_all<int, std::vector<std::vector<int>>>(delay);
    }
    void set_delay(std::vector<std::vector<int>> value) {
        update_matrix_variable_all<int, std::vector<std::vector<int>>>(delay, value);
    }
""",
        'init': """
        delay = init_matrix_variable<int, std::vector<std::vector<int>>>(1);
        update_matrix_variable_all<int>(delay, delays);

        idx_delay = 0;
        max_delay = %(pre_prefix)smax_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< std::vector< int > > > >(global_num_threads, std::vector< std::vector< std::vector< int > > >());
        for (int tid = 0; tid < global_num_threads; tid++) {
            _delayed_spikes[tid] = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(sub_matrices_[tid]->post_rank.size(), std::vector< int >() ) );
        }
    #ifdef _DEBUG
        std::cout << "Inited _delayed_spikes[" << global_num_threads << "][" << max_delay << "] and " << std::endl;
        for (int tid = 0; tid < global_num_threads; tid++) {
            std::cout << "   _delayed_spikes[" << tid << "][:] with vectors of size = " << _delayed_spikes[tid][0].size() << " element(s) " << std::endl;
        }
    #endif
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay = %(pre_prefix)smax_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< std::vector< int > > > >(global_num_threads, std::vector< std::vector< std::vector< int > > >());
        for (int tid = 0; tid < global_num_threads; tid++) {
            _delayed_spikes[tid] = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(sub_matrices_[tid]->post_rank.size(), std::vector< int >() ) );
        }

""",
        'pyx_struct':
"""
        # Non-uniform delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
        vector[vector[int]] get_delay()
        void set_delay(vector[vector[int]])
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.get_delay()
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.get_delay()[idx]
    def set_delay(self, value):
        proj%(id_proj)s.set_delay(value)
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    }
}

event_driven = {
    'declare': """
    std::vector< std::vector<std::vector<long> > > _last_event;
""",
    'cpp_init': """
        _last_event = init_matrix_variable<long, std::vector<std::vector<long> >>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

###############################################################
# Rate-coded continuous transmission using AVX
###############################################################
lil_summation_operation_avx_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            unsigned int _s, _stop;
            double _tmp_sum[4];

            std::vector<int>::size_type nb_post = sub_matrices_[tid]->post_rank.size();
            double* __restrict__ _pre_r = %(get_r)s;

            for (std::vector<%(idx_type)s>::size_type i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = sub_matrices_[tid]->pre_rank[i].data();
                _stop = sub_matrices_[tid]->pre_rank[i].size();

                __m256d _tmp_reg_sum = _mm256_set1_pd(0.0);

                _s = 0;
                for (; _s+8 < _stop; _s+=8) {
                    __m256d _tmp_r = _mm256_set_pd(
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256d _tmp_r2 = _mm256_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]
                    );

                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _tmp_r);
                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _tmp_r2);
                }
                _mm256_storeu_pd(_tmp_sum, _tmp_reg_sum);

                double lsum = 0.0;
                // partial sums
                for(char k = 0; k < 4; k++)
                    lsum += _tmp_sum[k];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                %(post_prefix)s_sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            unsigned int _s, _stop;
            float _tmp_sum[8];

            std::vector<int>::size_type nb_post = sub_matrices_[tid]->post_rank.size();
            float* __restrict__ _pre_r = %(get_r)s;

            for (std::vector<%(idx_type)s>::size_type i = 0; i < nb_post; i++) {
                int* __restrict__ _idx = sub_matrices_[tid]->pre_rank[i].data();

                _stop = sub_matrices_[tid]->pre_rank[i].size();
                __m256 _tmp_reg_sum = _mm256_set1_ps(0.0);
                __m256 _tmp_w = _mm256_set1_ps(w);
                __m256 _tmp_w2 = _mm256_set1_ps(w);

                _s = 0;
                for (; _s+16 < _stop; _s+=16) {
                    __m256 _tmp_r = _mm256_set_ps(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256 _tmp_r2 = _mm256_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]
                    );

                    _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _tmp_r);
                    _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _tmp_r2);
                }
                _mm256_storeu_ps(_tmp_sum, _tmp_reg_sum);

                float lsum = 0.0;
                // partial sums
                for(int k = 0; k < 8; k++)
                    lsum += _tmp_sum[k];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                %(post_prefix)s_sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """
    }
}

lil_summation_operation_avx = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            unsigned int _s, _stop;
            double _tmp_sum[4];

            std::vector<int>::size_type nb_post = sub_matrices_[tid]->post_rank.size();
            double* __restrict__ _pre_r = %(get_r)s;

            for (std::vector<%(idx_type)s>::size_type i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = sub_matrices_[tid]->pre_rank[i].data();
                double* __restrict__ _w = w[tid][i].data();
                _stop = sub_matrices_[tid]->pre_rank[i].size();

                __m256d _tmp_reg_sum = _mm256_set1_pd(0.0);

                _s = 0;
                for (; _s+8 < _stop; _s+=8) {
                    __m256d _tmp_r = _mm256_set_pd(
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256d _tmp_r2 = _mm256_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]
                    );

                    __m256d _tmp_w = _mm256_loadu_pd(&_w[_s]);
                    __m256d _tmp_w2 = _mm256_loadu_pd(&_w[_s+4]);

                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _mm256_mul_pd(_tmp_r, _tmp_w));
                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _mm256_mul_pd(_tmp_r2, _tmp_w2));
                }
                _mm256_storeu_pd(_tmp_sum, _tmp_reg_sum);

                double lsum = 0.0;
                // partial sums
                for(int k = 0; k < 4; k++)
                    lsum += _tmp_sum[k];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]] * _w[_s];

                %(post_prefix)s_sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """
    }
}

###############################################################
# Rate-coded continuous transmission (general case)
###############################################################
lil_summation_operation = {
    'sum' : """
%(pre_copy)s

    std::vector<%(idx_type)s>::size_type nb_post = sub_matrices_[tid]->post_rank.size();
    for(std::vector<%(idx_type)s>::size_type i = 0; i < nb_post; i++) {
        sum = 0.0;
        for(int j = 0; j < sub_matrices_[tid]->pre_rank[i].size(); j++) {
            sum += %(psp)s ;
        }
        %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
    }
"""
}

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables = {
    'local': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L) ){
    // Global variables
    %(global)s

    for(int i = 0; i < sub_matrices_[tid]->post_rank.size(); i++){
        rk_post = sub_matrices_[tid]->post_rank[i]; // Get postsynaptic rank

        // Semi-global variables
        %(semiglobal)s

        // Local variables
        for(int j = 0; j < sub_matrices_[tid]->pre_rank[i].size(); j++){
            rk_pre = sub_matrices_[tid]->pre_rank[i][j]; // Get presynaptic rank
    %(local)s
        }
    }
}
""",
    'global': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s

    auto post_rank = get_post_rank();
    // Local variables
    #pragma omp for
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
    %(semiglobal)s
    }

    post_rank.clear();
}
"""
}

###############################################################
# Spiking event-driven transmission
###############################################################
spiking_summation_fixed_delay = """
// Event-based summation
if (_transmission && %(post_prefix)s_active){

    // Iterate over all incoming spikes (possibly delayed constantly)
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++) {
        // Rank of the presynaptic neuron
        int rk_j = %(pre_array)s[_idx_j];

        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = sub_matrices_[tid]->inv_pre_rank.find(rk_j);
        if (inv_post_ptr == sub_matrices_[tid]->inv_pre_rank.end())
            continue;

        // List of postsynaptic neurons receiving spikes from that neuron
        std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
        // Number of post neurons
        int nb_post = inv_post.size();

        // Iterate over connected post neurons
        for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
            // Retrieve the correct indices
            int i = inv_post[_idx_i].first;
            int j = inv_post[_idx_i].second;

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""


# Uses a ring buffer to process non-uniform delays in spiking networks
spiking_summation_variable_delay = """
// Event-based summation
if (_transmission && %(post_prefix)s_active){

    // Iterate over the spikes emitted during the last step in the pre population
    for(int idx_spike=0; idx_spike<%(pre_prefix)sspiked.size(); idx_spike++){

        // Get the rank of the pre-synaptic neuron which spiked
        int rk_pre = %(pre_prefix)sspiked[idx_spike];
        // List of post neurons receiving connections
        auto rks_post_beg = (sub_matrices_[tid]->inv_pre_rank[rk_pre]).begin();
        auto rks_post_end = (sub_matrices_[tid]->inv_pre_rank[rk_pre]).end();

        // Iterate over the post neurons
        for (auto rks_post_it = rks_post_beg; rks_post_it != rks_post_end; rks_post_it++){
            // Index of the post neuron in the connectivity matrix
            int i = rks_post_it->first ;
            // Index of the pre neuron in the connecivity matrix
            int j = rks_post_it->second ;
            // Delay of that connection
            int d = delay[tid][i][j]-1;
            // Index in the ring buffer
            int modulo_delay = (idx_delay + d) %% max_delay;
            // Add the spike in the ring buffer
            _delayed_spikes[tid][modulo_delay][i].push_back(j);
        }
    }

    #pragma omp barrier

    // Iterate over all post neurons having received spikes in the previous steps
    for (int i=0; i<_delayed_spikes[tid][idx_delay].size(); i++){

        for (int _idx_j=0; _idx_j<_delayed_spikes[tid][idx_delay][i].size(); _idx_j++){
            // Pre-synaptic index in the connectivity matrix
            int j = _delayed_spikes[tid][idx_delay][i][_idx_j];

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
        // Empty the current list of the ring buffer
        _delayed_spikes[tid][idx_delay][i].clear();
    }
    #pragma omp barrier

    // Increment the index of the ring buffer
    #pragma omp single
    {
        idx_delay = (idx_delay + 1) %% max_delay;
    }
} // active
"""

spiking_post_event = """
if(_transmission && %(post_prefix)s_active){

    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++){
        // In which sub matrix the neuron take place
        int rk_post = %(post_prefix)sspiked[_idx_i];

        // Find its index in the projection
        auto it = find(sub_matrices_[tid]->post_rank.begin(), sub_matrices_[tid]->post_rank.end(), rk_post);

        // Leave if the neuron is not part of the projection
        if (it==sub_matrices_[tid]->post_rank.end()) continue;

        // which position
        int i = std::distance(sub_matrices_[tid]->post_rank.begin(), it);

        // Iterate over all synapse to this neuron
        int nb_pre = sub_matrices_[tid]->pre_rank[i].size();
        for(int j = 0; j < nb_pre; j++){
%(event_driven)s
%(post_event)s
        }
    }
}
"""

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,
    'event_driven': event_driven,
    'rng_update': cpp_11_rng,

    # operations
    'rate_coded_sum': lil_summation_operation,
    'vectorized_default_psp': {
        'avx': {
            'single_w': lil_summation_operation_avx_single_weight,
            'multi_w': lil_summation_operation_avx
        }
    },
    'update_variables': update_variables,
    'spiking_sum_fixed_delay': {
        'outer_loop': spiking_summation_fixed_delay
    },
    'spiking_sum_variable_delay': spiking_summation_variable_delay,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': "[tid][i][j]",
    'semiglobal_index': '[tid][i]',
    'global_index': '',
    'pre_index': '[sub_matrices_[tid]->pre_rank[i][j]]',
    'post_index': '[sub_matrices_[tid]->post_rank[i]]',
    'delay_nu' : '[delay[tid][i][j]-1]', # non-uniform delay
    'delay_u' : '[delay-1]' # uniform delay
}
