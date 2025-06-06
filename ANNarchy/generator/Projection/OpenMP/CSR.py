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
        %(name)s = init_matrix_variable(static_cast<%(float_prec)s>(%(init)s));
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>( post_ranks_.size(), %(init)s);
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

delay = {
    # A single value for all synapses
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;

    int get_delay() { return delay; }
    int get_dendrite_delay(int idx) { return delay; }
    void set_delay(int delay) { this->delay = delay; }
""",
        'init': """
    delay = delays[0][0];
"""
    },
    # An individual value for each synapse
    'nonuniform_rate_coded': {
        'declare': """
    std::vector<int> delay;
    int max_delay;

    std::vector<std::vector<int>> get_delay() { return get_matrix_variable_all<int>(delay); }
    void set_delay(std::vector<std::vector<int>> value) { update_matrix_variable_all<int>(delay, value); }
    std::vector<int> get_dendrite_delay(int lil_idx) { return get_matrix_variable_row<int>(delay, lil_idx); }
    int get_max_delay() { return max_delay; }
    void set_max_delay() { this->max_delay = max_delay; }
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    max_delay = %(pre_prefix)smax_delay;
""",
        'reset': ""
    },
    # An individual value for each synapse
    'nonuniform_spiking': {
        'declare': """
    std::vector<int> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes;

    std::vector<std::vector<int>> get_delay() { return get_matrix_variable_all<int>(delay); }
    void set_delay(std::vector<std::vector<int>> value) { update_matrix_variable_all<int>(delay, value); }
    std::vector<int> get_dendrite_delay(int lil_idx) { return get_matrix_variable_row<int>(delay, lil_idx); }
    int get_max_delay() { return max_delay; }
    void set_max_delay() { this->max_delay = max_delay; }
""",
        'init': """
    delay = init_matrix_variable<int>(1);
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

event_driven = {
    'declare': """
    std::vector<long> _last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

###############################################################################
# Rate-coded continuous transmission
###############################################################################
csr_summation_operation = {
    'sum' : """
%(pre_copy)s

// w as CSR
const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
const %(idx_type)s* __restrict__ col_idx = col_idx_.data();
%(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
%(float_prec)s* __restrict__ target_ptr = %(post_prefix)s_sum_%(target)s.data();

%(omp_code)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s rk_post = post_ranks_[i];
    %(float_prec)s sum = 0.0;
    for (%(size_type)s j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++) {
        sum += %(psp)s;
    }
    target_ptr%(post_index)s += sum;
} 
""",
    'max': """
%(pre_copy)s
nb_post = post_rank.size();

%(omp_code)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s rk_post = post_ranks_[i];
    %(size_type)s j = _row_ptr[i];
    sum = %(psp)s ;

    for (%(size_type)s j = _row_ptr[rk_post]+1; j < _row_ptr[rk_post+1]; j++) {
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
""",
    'min': """
%(pre_copy)s
nb_post = post_rank.size();

%(omp_code)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s rk_post = post_ranks_[i];
    %(size_type)s j = _row_ptr[i];
    sum = %(psp)s ;

    for (%(size_type)s j = _row_ptr[rk_post]+1; j < _row_ptr[rk_post+1]; j++) {
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
""",
    'mean': """
%(pre_copy)s
nb_post = post_rank.size();

%(omp_code)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(size_type)s j = _row_ptr[i];
    sum = 0.0 ;

    for (%(size_type)s j = _row_ptr[rk_post]; j < _row_ptr[rk_post+1]; j++){
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum / static_cast<%(float_prec)s>(pre_rank[i].size());
}
"""
}

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using
# SIMD instructions and single weight value for all synapses in the projection.
###############################################################################
continuous_transmission_sse_single_weight = {
    'sum' : {
        'double': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s->_active) {
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();

            double _tmp_sum[2];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m128d _tmp_reg_sum = _mm_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
                    __m128d _tmp_r = _mm_set_pd(_pre_r[_idx[_s+1]], _pre_r[_idx[_s]]);
                    __m128d _tmp_r2 = _mm_set_pd(_pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]]);
                    __m128d _tmp_r3 = _mm_set_pd(_pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]);
                    __m128d _tmp_r4 = _mm_set_pd(_pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]]);

                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _tmp_r);
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _tmp_r2);
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _tmp_r3);
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _tmp_r4);
                }
                _mm_storeu_pd(_tmp_sum, _tmp_reg_sum);

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #endif
""",
        'float': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s->_active) {
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();

            float _tmp_sum[4];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m128 _tmp_reg_sum = _mm_setzero_ps();

                for (; (_s+16) < _stop; _s+=16) {
                    __m128 _tmp_r = _mm_set_ps(_pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]);
                    __m128 _tmp_r2 = _mm_set_ps(_pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]);
                    __m128 _tmp_r3 = _mm_set_ps(_pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]);
                    __m128 _tmp_r4 = _mm_set_ps(_pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]]);

                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _tmp_r);
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _tmp_r2);
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _tmp_r3);
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _tmp_r4);
                }
                _mm_storeu_ps(_tmp_sum, _tmp_reg_sum);

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

continuous_transmission_avx_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];

                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m256d _tmp_reg_sum = _mm256_setzero_pd();

                for (; _s+8 < _stop; _s+=8) {
                    __m256d _tmp_r = _mm256_set_pd(
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256d _tmp_r2 = _mm256_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]
                    );

                    _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _mm256_add_pd(_tmp_r, _tmp_r2));
                }
                _mm256_storeu_pd(_tmp_sum, _tmp_reg_sum);

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

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
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];

                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m256 _tmp_reg_sum = _mm256_setzero_ps();

                for (; (_s+16) < _stop; _s+=16) {
                    __m256 _tmp_r = _mm256_set_ps(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m256 _tmp_r2 = _mm256_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]
                    );

                    _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _mm256_add_ps(_tmp_r, _tmp_r2));
                }
                _mm256_storeu_ps(_tmp_sum, _tmp_reg_sum);

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

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

continuous_transmission_avx512_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();

        if (_transmission && %(post_prefix)s_active) {
            double _tmp_sum[8];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m512d _tmp_reg_sum = _mm512_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
                    __m512d _tmp_r = _mm512_set_pd(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );

                    _tmp_reg_sum = _mm512_add_pd(_tmp_reg_sum, _tmp_r);
                }

                _mm512_storeu_pd(_tmp_sum, _tmp_reg_sum);

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                %(post_prefix)s_sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
""",
        'float': """
    #ifdef __AVX__
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();

        if (_transmission && %(post_prefix)s_active) {
            float _tmp_sum[16];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];

                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m512 _tmp_reg_sum = _mm512_setzero_ps();

                for (; (_s+16) < _stop; _s+=16) {
                    __m512 _tmp_r = _mm512_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]],
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );

                    _tmp_reg_sum = _mm512_add_ps(_tmp_reg_sum, _tmp_r);
                }
                _mm512_storeu_ps(_tmp_sum, _tmp_reg_sum);

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7] + _tmp_sum[8] + _tmp_sum[9] + _tmp_sum[10] + _tmp_sum[11] + _tmp_sum[12] + _tmp_sum[13] + _tmp_sum[14] + _tmp_sum[15];

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

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using 
# SIMD instructions
###############################################################################
continuous_transmission_sse = {
    'sum' : {
        'double': """
    #ifdef __SSE4_1__

        if (_transmission && %(post_prefix)s_active) {
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();
            const double* __restrict__ _w = w.data();

            double _tmp_sum[2];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m128d _tmp_reg_sum = _mm_setzero_pd();

                for (; _s+8 < _stop; _s+=8) {
                    __m128d _tmp_r = _mm_set_pd(_pre_r[_idx[_s+1]], _pre_r[_idx[_s+0]]);
                    __m128d _tmp_r2 = _mm_set_pd(_pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]]);
                    __m128d _tmp_r3 = _mm_set_pd(_pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]);
                    __m128d _tmp_r4 = _mm_set_pd(_pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]]);

                    __m128d _tmp_w = _mm_loadu_pd(&_w[_s]);
                    __m128d _tmp_w2 = _mm_loadu_pd(&_w[_s+2]);
                    __m128d _tmp_w3 = _mm_loadu_pd(&_w[_s+4]);
                    __m128d _tmp_w4 = _mm_loadu_pd(&_w[_s+6]);

                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _mm_mul_pd(_tmp_r, _tmp_w));
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _mm_mul_pd(_tmp_r2, _tmp_w2));
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _mm_mul_pd(_tmp_r3, _tmp_w3));
                    _tmp_reg_sum = _mm_add_pd(_tmp_reg_sum, _mm_mul_pd(_tmp_r4, _tmp_w4));
                }

                _mm_storeu_pd(_tmp_sum, _tmp_reg_sum);

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1];

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
    #ifdef __SSE4_1__
        if (_transmission && %(post_prefix)s_active) {
            const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
            const %(idx_type)s* __restrict__ _idx = col_idx_.data();
            const float* __restrict__ _w = w.data();

            float _tmp_sum[4];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];

                _s = row_ptr[rk_post];
                _stop = row_ptr[rk_post+1];
                __m128 _tmp_reg_sum = _mm_setzero_ps();

                for (; _s+16 < _stop; _s+=16) {
                    __m128 _tmp_r = _mm_set_ps(_pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]);
                    __m128 _tmp_r2 = _mm_set_ps(_pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]]);
                    __m128 _tmp_r3 = _mm_set_ps(_pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]);
                    __m128 _tmp_r4 = _mm_set_ps(_pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]]);

                    __m128 _tmp_w = _mm_loadu_ps(&_w[_s]);
                    __m128 _tmp_w2 = _mm_loadu_ps(&_w[_s+4]);
                    __m128 _tmp_w3 = _mm_loadu_ps(&_w[_s+8]);
                    __m128 _tmp_w4 = _mm_loadu_ps(&_w[_s+12]);

                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r, _tmp_w));
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r2, _tmp_w2));
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r3, _tmp_w3));
                    _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r4, _tmp_w4));
                }
                _mm_storeu_ps(_tmp_sum, _tmp_reg_sum);

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]] * _w[_s];

                %(post_prefix)s_sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

continuous_transmission_avx = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();
        const double* __restrict__ _w = w.data();

        if (_transmission && pop%(id_post)s->_active) {
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];

                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m256d _tmp_reg_sum = _mm256_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
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

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

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
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();
        const float* __restrict__ _w = w.data();

        if (_transmission && %(post_prefix)s_active) {
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];

                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m256 _tmp_reg_sum = _mm256_setzero_ps();

                for (; (_s+16) < _stop; _s+=16) {
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

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

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
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();
        const double* __restrict__ _w = w.data();

        if (_transmission && %(post_prefix)s_active) {
            double _tmp_sum[8];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
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

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]] * _w[_s];

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
""",
        'float': """
    #ifdef __AVX512F__
        const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
        const %(idx_type)s* __restrict__ _idx = col_idx_.data();
        const float* __restrict__ _w = w.data();

        if (_transmission && pop%(id_post)s->_active) {
            float _tmp_sum[16];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_ranks_.size(); i++) {
                %(idx_type)s rk_post = post_ranks_[i];
                %(size_type)s _s = row_ptr[rk_post];
                %(size_type)s _stop = row_ptr[rk_post+1];
                __m512 _tmp_reg_sum = _mm512_setzero_ps();

                for (; (_s+16) < _stop; _s+=16) {
                    __m512 _tmp_r = _mm512_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]],
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );

                    __m512 _tmp_w = _mm512_loadu_ps(&_w[_s]);

                    _tmp_reg_sum = _mm512_add_ps(_tmp_reg_sum, _mm512_mul_ps(_tmp_r, _tmp_w));
                }
                _mm512_storeu_ps(_tmp_sum, _tmp_reg_sum);

                // partial sums
                float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7] + _tmp_sum[8] + _tmp_sum[9] + _tmp_sum[10] + _tmp_sum[11] + _tmp_sum[12] + _tmp_sum[13] + _tmp_sum[14] + _tmp_sum[15];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]] * _w[_s];

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

update_variables = {
    'local': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s

    const %(size_type)s* __restrict__ row_ptr = row_begin_.data();
    const %(idx_type)s* __restrict__ col_idx = col_idx_.data();
    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());

    #pragma omp for
    for(int i = 0; i < nb_post; i++){
        rk_post = post_ranks_[i];
    %(semiglobal)s
        for(int j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++){
            rk_pre = col_idx[j];
    %(local)s
        }
    }
}
""",
        'global': """
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s

    %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_ranks_.size());
    %#pragma omp for
    for(%(idx_type)s i = 0; i < nb_post; i++){
        rk_post = post_ranks[i];
    %(semiglobal)s
    }
}
"""
}

###############################################################################
# Event-based computation
###############################################################################
spiking_summation_fixed_delay_inner_loop = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        int _pre = %(pre_array)s[_idx];

        // Iterate over connected post neurons
        #pragma omp for
        for (int syn = _col_ptr[_pre]; syn < _col_ptr[_pre + 1]; syn++) {
            %(event_driven)s
            %(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_summation_fixed_delay_outer_loop = """// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    for( int _idx = tid; _idx < %(pre_array)s.size(); _idx += nt) {
        int _pre = %(pre_array)s[_idx];

        // Iterate over connected post neurons
        for (int syn = _col_ptr[_pre]; syn < _col_ptr[_pre + 1]; syn++) {
            %(event_driven)s
            #pragma omp atomic%(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_post_event = """
// w as CSR
const int * __restrict__ row_ptr = row_begin_.data();
const int * __restrict__ _col_idx = col_idx_.data();

if(_transmission && %(post_prefix)s_active){
    #pragma omp for
    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = %(post_prefix)sspiked[_idx_i];

        // Iterate over all synapse to this neuron
        for (int j = row_ptr[rk_post]; j < row_ptr[rk_post+1]; j++) {
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

    # operations
    'rate_coded_sum': csr_summation_operation,
    'vectorized_default_psp': {
        'sse': {
            'single_w': continuous_transmission_sse_single_weight,
            'multi_w': continuous_transmission_sse
        },
        'avx': {
            'single_w': continuous_transmission_avx_single_weight,
            'multi_w': continuous_transmission_avx
        },
        'avx512': {
            'single_w': continuous_transmission_avx512_single_weight,
            'multi_w': continuous_transmission_avx512
        }
    },
    'update_variables': update_variables,
    'spiking_sum_fixed_delay': {
        'inner_loop': spiking_summation_fixed_delay_inner_loop,
        'outer_loop': spiking_summation_fixed_delay_outer_loop
    },
    'spiking_sum_variable_delay': None,
    'post_event': spiking_post_event
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[i]',
    'global_index': '',
    'pre_index': '[col_idx[j]]',
    'post_index': '[post_ranks_[i]]',
    'delay_nu' : '[delay[j]-1]', # non-uniform delay
    'delay_u' : '[delay-1]' # uniform delay
}
