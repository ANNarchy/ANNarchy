"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
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
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
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
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global
        size_in_bytes += sizeof(%(ctype)s);
"""
}

attribute_cpp_delete = {
    'local': """
        // %(name)s
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

#############################################
##  Synaptic delay
#############################################
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
    std::vector<int> delay;
    int max_delay;

    std::vector<std::vector<int>> get_delay() { return get_matrix_variable_all<int>(delay); }
    void set_delay(std::vector<std::vector<int>> value) { update_matrix_variable_all<int>(delay, value); }
    std::vector<int> get_dendrite_delay(int lil_idx) { return get_matrix_variable_row<int>(delay, lil_idx); }
""",
        'init': """
    delay = init_variable<int>(1);
    update_variable_all<int>(delay, delays);
""",
        'reset': "",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] get_delay()
        void set_delay(vector[vector[int]])
        vector[int] get_dendrite_delay(int)
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
    std::vector<int> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes;
""",
        'init': """
    delay = init_variable<int>(1);
    update_variable_all<int>(delay, delays);

    idx_delay = 0;
    max_delay = %(pre_prefix)smax_delay;
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay =  %(pre_prefix)smax_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );
""",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
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

###############################################################
# Rate-coded continuous transmission
###############################################################
ellr_summation_operation = {
    'sum' : """
%(pre_copy)s

%(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s rk_post = post_ranks_[i]; // Get postsynaptic rank

    sum = 0.0;
    for(%(size_type)s j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++) {
        %(idx_type)s rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}"""
}

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using AVX
#
# For details on single_weight: see lil_summation_operation_avx_single_weight
###############################################################################
continuous_transmission_avx_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            %(size_type)s _s, _stop;
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;

            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(idx_type)s* __restrict__ _idx = col_idx_.data();

                _s = i*maxnzr_;
                _stop = i*maxnzr_+rl_[i];
                __m256d _tmp_reg_sum = _mm256_setzero_pd();

                for (; _s+8 < _stop; _s+=8) {
                    __m256d _tmp_r1 = _mm256_set_pd(
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256d _tmp_r2 = _mm256_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]
                    );

                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _tmp_r1);
                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _tmp_r2);
                }

                _mm256_storeu_pd(_tmp_sum, _tmp_reg_sum);
                double lsum = static_cast<double>(0.0);
                // partial sums
                for(int k = 0; k < 4; k++)
                    lsum += _tmp_sum[k];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                %(post_prefix)s_sum_%(target)s%(post_index)s += lsum * w;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using AVX
###############################################################################
continuous_transmission_avx = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            %(size_type)s _s, _stop;
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;

            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(idx_type)s* __restrict__ _idx = col_idx_.data();
                double* __restrict__ _w = w.data();

                _s = i*maxnzr_;
                _stop = i*maxnzr_+rl_[i];
                __m256d _tmp_reg_sum = _mm256_setzero_pd();

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
                double lsum = static_cast<double>(0.0);
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
""",
        'float': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            %(size_type)s _s, _stop;
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;

            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(idx_type)s* __restrict__ _idx = col_idx_.data();
                float* __restrict__ _w = w.data();

                _s = i*maxnzr_;
                _stop = i*maxnzr_+rl_[i];
                __m256 _tmp_reg_sum = _mm256_setzero_ps();

                for (; _s+16 < _stop; _s+=16) {
                    __m256 _tmp_r = _mm256_set_ps(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256 _tmp_r2 = _mm256_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]
                    );

                    __m256 _tmp_w = _mm256_loadu_ps(&_w[_s]);
                    __m256 _tmp_w2 = _mm256_loadu_ps(&_w[_s+8]);

                    _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _mm256_mul_ps(_tmp_r, _tmp_w));
                    _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _mm256_mul_ps(_tmp_r2, _tmp_w2));
                }

                _mm256_storeu_ps(_tmp_sum, _tmp_reg_sum);
                float lsum = static_cast<double>(0.0);
                // partial sums
                for (int k = 0; k < 8; k++)
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

continuous_transmission_avx512 = {
    'sum' : {
        'double': """
    #ifdef __AVX512F__
        if (_transmission && %(post_prefix)s_active) {
            %(idx_type)s* __restrict__ _idx = col_idx_.data();
            const double* __restrict__ _w = w.data();

            double _tmp_sum[8];
            const double* __restrict__ _pre_r = %(get_r)s;

            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = i*maxnzr_;
                %(size_type)s _stop = i*maxnzr_+rl_[i];
                __m512d _tmp_reg_sum = _mm512_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
                    __m512d _tmp_r = _mm512_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );

                    __m512d _tmp_w = _mm512_loadu_pd(&_w[_s]);

                    _tmp_reg_sum = _mm512_add_pd(_tmp_reg_sum, _mm512_mul_pd(_tmp_r, _tmp_w));
                }

                _mm512_storeu_pd(_tmp_sum, _tmp_reg_sum);

                double lsum = static_cast<double>(0.0);
                // partial sums
                for(int k = 0; k < 8; k++)
                    lsum += _tmp_sum[k];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]] * _w[_s];

                %(post_prefix)s_sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
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

    // Local variables
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
    for (%(idx_type)s i = 0; i < nb_post; i++) {
        rk_post = post_ranks_[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s

        // Local variables
        for(%(size_type)s j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++){
            rk_pre = col_idx_[j]; // Get presynaptic rank
    %(local)s
        }
    }
}
"""
}

conn_templates = {
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'delay': delay,
    
    'rate_coded_sum': ellr_summation_operation,
    'vectorized_default_psp': {
        'avx': {
            'single_w': continuous_transmission_avx_single_weight,
            'multi_w': continuous_transmission_avx
        },
        'avx512': {
            'multi_w': continuous_transmission_avx512
        }
    },
    'update_variables': update_variables
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}
