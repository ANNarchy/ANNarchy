"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s ;
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
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
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
    for (%(idx_type)s i = 0; i < post_rank.size(); i++) {
    %(semiglobal_rng)s
        for (%(idx_type)s j = 0; j < pre_rank[i].size(); j++) {
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
""",
        'reset': ""
    },
    # An individual value for each synapse
    'nonuniform_rate_coded': {
        'declare': """
    std::vector<std::vector<int>> delay;
    int max_delay;

    std::vector<std::vector<int>> get_delay() { return delay; }
    std::vector<int> get_dendrite_delay(int idx) {
        if (idx < delay.size()) {
            return delay[idx];
        } else {
            std::cerr << "ProjStruct%(id)s::get_dendrite_delay(): invalid idx used." << std::endl;
            return std::vector<int>();
        }
    }
    void set_delay(std::vector<std::vector<int>> delay) {
        this->delay = delay;
    }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int max_delay) { this->max_delay = max_delay; }
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    max_delay = %(pre_prefix)smax_delay;
""",
        'reset': ""
    },
    # An individual value for each synapse and a
    # buffer for spike events
    'nonuniform_spiking': {
        'declare': """
    std::vector<std::vector<int>> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes;

    std::vector<std::vector<int>> get_delay() { return get_matrix_variable_all<int>(delay); }
    void set_delay(std::vector<std::vector<int>> value) { update_matrix_variable_all<int>(delay, value); }
    std::vector<int> get_dendrite_delay(int lil_idx) { return get_matrix_variable_row<int>(delay, lil_idx); }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int max_delay) { this->max_delay = max_delay; }
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    idx_delay = 0;
    max_delay = %(pre_prefix)smax_delay ;
    _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay = %(pre_prefix)smax_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );
"""
    }
}

event_driven = {
    'declare': """
    std::vector< std::vector<long> > _last_event;
""",
    'cpp_init': """
        _last_event = init_matrix_variable<long>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

###############################################################
# Rate-coded continuous transmission (general case)
###############################################################
lil_summation_operation = {
    'sum' : """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

%(omp_code)s %(omp_clause)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    sum = 0.0;
    nb_pre = static_cast<%(idx_type)s>(pre_rank[i].size());
    for (%(idx_type)s j = 0; j < nb_pre; j++) {
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
""",
    'max': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

%(omp_code)s %(omp_clause)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s j = 0;
    sum = %(psp)s ;
    for (j = 1; j < pre_rank[i].size(); j++) {
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
""",
    'min': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

%(omp_code)s %(omp_clause)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s j = 0;
    sum = %(psp)s ;
    for (j = 1; j < pre_rank[i].size(); j++) {
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum;
}
""",
    'mean': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

%(omp_code)s %(omp_clause)s %(omp_schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    sum = 0.0 ;
    for(%(idx_type)s j = 0; j < pre_rank[i].size(); j++) {
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s%(post_index)s += sum / static_cast<%(float_prec)s>(pre_rank[i].size());
}
"""
}

###############################################################
# Rate-coded continuous transmission using SIMD instructions
# and a single weight
###############################################################
lil_summation_operation_sse_single_weight = {
    'sum' : {
        'double': """
    #ifdef __SSE4_1__
        if (_transmission && %(post_prefix)s_active) {
            %(idx_type)s _s, _stop;
            double _tmp_sum[2];
            double* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                _stop = pre_rank[i].size();

                __m128d _tmp_reg_sum = _mm_set1_pd(0.0);
                _s = 0;
                for (; _s+8 < _stop; _s+=8) {
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

                %(post_prefix)s_sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #endif
""",
        'float': """
    #ifdef __SSE4_1__
        if (_transmission && %(post_prefix)s_active) {
            %(idx_type)s _s, _stop;
            float _tmp_sum[4];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                _stop = pre_rank[i].size();

                __m128 _tmp_reg_sum = _mm_set1_ps(0.0);
                _s = 0;
                for (; _s+16 < _stop; _s+=16) {
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

                %(post_prefix)s_sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

lil_summation_operation_avx_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && pop%(id_post)s->_active) {
            double* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = static_cast<%(idx_type)s>(pre_rank[i].size());
                double _tmp_sum[4];
                __m256d _tmp_reg_sum = _mm256_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
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
            %(idx_type)s _s, _stop;
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                _stop = pre_rank[i].size();

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
    #ifdef __AVX512F__
        if (_transmission && pop%(id_post)s->_active) {
            double _tmp_sum[8];
            double* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_rank.size(); i++) {
                %(idx_type)s rk_post = post_rank[i];
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = static_cast<%(idx_type)s>(pre_rank[i].size());

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

                pop%(id_post)s._sum_%(target)s%(post_index)s +=  w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __AVX512F__
        if (_transmission && pop%(id_post)s->_active) {
            float _tmp_sum[16];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for firstprivate(w)
            for (%(idx_type)s i = 0; i < post_rank.size(); i++) {
                %(idx_type)s rk_post = post_rank[i];
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = static_cast<%(idx_type)s>(pre_rank[i].size());
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

                pop%(id_post)s._sum_%(target)s%(post_index)s +=  w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
    """
    }
}

###############################################################
# Rate-coded continuous transmission using SIMD instructions
###############################################################
continuous_transmission_sse = {
    'sum' : {
        'double': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s->_active) {
            double* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                double* __restrict__ _w = w[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = static_cast<%(idx_type)s>(pre_rank[i].size());
                double _tmp_sum[2];
                __m128d _tmp_reg_sum = _mm_setzero_pd();

                for (; (_s+8) < _stop; _s+=8) {
                    __m128d _tmp_r = _mm_set_pd(_pre_r[_idx[_s+1]], _pre_r[_idx[_s]]);
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s->_active) {
            , _stop;
            float _tmp_sum[4];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                float* __restrict__ _w = w[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = static_cast<%(idx_type)s>(pre_rank[i].size());
                __m128 _tmp_reg_sum = _mm_setzero_pd();

                for (; (_s+16) < _stop; _s+=16) {
                    __m128 _tmp_r = _mm_set_ps(
                        _pre_r[_idx[_s+3]], _pre_r[_idx[_s+2]], _pre_r[_idx[_s+1]], _pre_r[_idx[_s]]
                    );
                    __m128 _tmp_r = _mm_set_ps(
                        _pre_r[_idx[_s+7]], _pre_r[_idx[_s+6]], _pre_r[_idx[_s+5]], _pre_r[_idx[_s+4]],
                    );
                    __m128 _tmp_r = _mm_set_ps(
                        _pre_r[_idx[_s+11]], _pre_r[_idx[_s+10]], _pre_r[_idx[_s+9]], _pre_r[_idx[_s+8]]
                    );
                    __m128 _tmp_r = _mm_set_ps(
                        _pre_r[_idx[_s+15]], _pre_r[_idx[_s+14]], _pre_r[_idx[_s+13]], _pre_r[_idx[_s+12]],
                    );

                    __m128 _tmp_w = _mm_loadu_ps(&_w[_s]);
                    __m128 _tmp_w2 = _mm_loadu_ps(&_w[_s+4]);
                    __m128 _tmp_w3 = _mm_loadu_ps(&_w[_s+8]);
                    __m128 _tmp_w3 = _mm_loadu_ps(&_w[_s+12]);

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

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """
    }
}

continuous_transmission_avx = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && %(post_prefix)s_active) {
            %(idx_type)s _s, _stop;
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                double* __restrict__ _w = w[i].data();

                _stop = pre_rank[i].size();

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
        if (_transmission && %(post_prefix)s_active) {
            %(idx_type)s _s, _stop;
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                int* __restrict__ _idx = pre_rank[i].data();
                float* __restrict__ _w = w[i].data();

                _stop = pre_rank[i].size();
                __m256 _tmp_reg_sum = _mm256_set1_ps(0.0);

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
        if (_transmission && pop%(id_post)s->_active) {
            double* __restrict__ _pre_r = %(get_r)s;
            double _tmp_sum[8];

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_rank.size(); i++) {
                %(idx_type)s rk_post = post_rank[i];
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = pre_rank[i].size();
                double* __restrict__ _w = w[i].data();
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

                pop%(id_post)s._sum_%(target)s%(post_index)s +=  lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __AVX512F__
        if (_transmission && pop%(id_post)s->_active) {
            float _tmp_sum[16];
            float* __restrict__ _pre_r = %(get_r)s;

            #pragma omp for
            for (%(idx_type)s i = 0; i < post_rank.size(); i++) {
                %(idx_type)s rk_post = post_rank[i];
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                %(idx_type)s _s = 0;
                %(idx_type)s _stop = pre_rank[i].size();
                float* __restrict__ _w = w[i].data();
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

                pop%(id_post)s._sum_%(target)s%(post_index)s +=  lsum;
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

    #pragma omp for private(rk_post, rk_pre) firstprivate(dt)
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s
        // Local variables
        for(int j = 0; j < pre_rank[i].size(); j++){
            rk_pre = pre_rank[i][j]; // Get presynaptic rank
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

    // Local variables
    #pragma omp for
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
    %(semiglobal)s
    }
}
"""
}

###############################################################
# Spiking event-driven transmission
###############################################################
spiking_summation_fixed_delay_outer_loop = """
// Event-based summation
if (_transmission && %(post_prefix)s_active){

    // Iterate over all incoming spikes (possibly delayed constantly)
    for(int _idx_j = tid; _idx_j < %(pre_array)s.size(); _idx_j += nt) {
        // Rank of the presynaptic neuron
        int rk_j = %(pre_array)s[_idx_j];

        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = inv_pre_rank.find(rk_j);
        if (inv_post_ptr == inv_pre_rank.end())
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
            #pragma omp atomic%(g_target)s

            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""

spiking_summation_fixed_delay_inner_loop = """
// Event-based summation
if (_transmission && %(post_prefix)s_active) {

    %(spiked_array_fusion)s

    // Iterate over all incoming spikes (possibly delayed constantly)
    for (int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++ ) {
        // Rank of the presynaptic neuron
        int rk_j = %(pre_array)s[_idx_j];

        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = inv_pre_rank.find(rk_j);
        if (inv_post_ptr == inv_pre_rank.end())
            continue;

        // List of postsynaptic neurons receiving spikes from that neuron
        std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;

        // Number of post neurons
        int nb_post = inv_post.size();

        // Iterate over connected post neurons
        #pragma omp for
        for (int _idx_i = 0; _idx_i < nb_post; _idx_i++){
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
    #pragma omp for
    for(int idx_spike=0; idx_spike<%(pre_prefix)sspiked.size(); idx_spike++){

        // Get the rank of the pre-synaptic neuron which spiked
        int rk_pre = %(pre_prefix)sspiked[idx_spike];

        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = inv_pre_rank.find(rk_pre);
        if (inv_post_ptr == inv_pre_rank.end())
            continue;

        // List of post neurons receiving connections from rk_pre
        std::vector< std::pair<int, int> >& rks_post = inv_post_ptr->second;

        // Iterate over the post neurons
        for(int x=0; x<rks_post.size(); x++){
            // Index of the post neuron in the connectivity matrix
            int i = rks_post[x].first ;
            // Index of the pre neuron in the connecivity matrix
            int j = rks_post[x].second ;
            // Delay of that connection
            int d = delay[i][j]-1;
            // Index in the ring buffer
            int modulo_delay = (idx_delay + d) %% max_delay;
            // Add the spike in the ring buffer
            #pragma omp critical
            {
                _delayed_spikes[modulo_delay][i].push_back(j);
            }
        }
    }

    #pragma omp barrier

    // Iterate over all post neurons having received spikes in the previous steps
    for (int i=0; i<_delayed_spikes[idx_delay].size(); i++){
        #pragma omp for
        for (int _idx_j=0; _idx_j<_delayed_spikes[idx_delay][i].size(); _idx_j++){
            // Pre-synaptic index in the connectivity matrix
            int j = _delayed_spikes[idx_delay][i][_idx_j];

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            #pragma omp critical
            {
            %(g_target)s
            }
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
        // Empty the current list of the ring buffer
        #pragma omp single
        {
            _delayed_spikes[idx_delay][i].clear();
        }
    }

    #pragma omp barrier

    #pragma omp single
    {
        // Increment the index of the ring buffer
        idx_delay = (idx_delay + 1) %% max_delay;
    }
} // active
"""

spiking_post_event = """
if(_transmission && %(post_prefix)s_active){

    #pragma omp for
    for(int _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++){
        // In which sub matrix the neuron take place
        int rk_post = %(post_prefix)sspiked[_idx_i];

        // Find its index in the projection
        auto it = find(post_rank.begin(), post_rank.end(), rk_post);

        // Leave if the neuron is not part of the projection
        if (it==post_rank.end()) continue;

        // which position
        int i = std::distance(post_rank.begin(), it);

        // Iterate over all synapse to this neuron
        int nb_pre = pre_rank[i].size();
        for(int j = 0; j < nb_pre; j++){
%(event_driven)s
%(post_event)s
        }
    }
}
"""

######################################
### Structural plasticity
######################################
# All code templates needed for structural plasticity.
structural_plasticity = {
    'header_struct': {
        'header': """
    /*
     * Structural plasticity
     */
    bool synapse_exists(int post_idx, int pre_rk) {
        // Check if a synapse to neuron *pre_rk* exists
        return std::binary_search(pre_rank[post_idx].begin(), pre_rank[post_idx].end(), pre_rk);
    }
    void _add_at_position(int post_idx, int pos, int pre_rk, %(float_prec)s _weight, int _delay=0%(extra_args)s) {
        pre_rank[post_idx].insert(pre_rank[post_idx].begin() + pos, pre_rk);
        w[post_idx].insert(w[post_idx].begin() + pos, _weight);

        // Update additional fields
%(delay_code)s
%(add_code)s
%(spike_add)s
%(rd_add)s
    }
    void _erase_at_position(int post_idx, int pos) {
        // Update connectivty
        pre_rank[post_idx].erase(pre_rank[post_idx].begin() + pos);
        w[post_idx].erase(w[post_idx].begin() + pos);

        // Update additional fields
%(delay_remove)s
%(add_remove)s
%(spike_remove)s
%(rd_remove)s
    }
    void add_single_synapse(int post_idx, int pre, %(float_prec)s _weight, int _delay=0%(extra_args)s) {
        // Find where to put the synapse
        int idx = pre_rank[post_idx].size();
        for(int i=0; i<pre_rank[post_idx].size(); i++){
            if(pre_rank[post_idx][i] > pre){
                idx = i;
                break;
            }
        }

        // Update connectivty
        _add_at_position(post_idx, idx, pre, _weight, _delay%(extra_args_acc)s);
    }
    void add_multiple_synapses(int post_idx, std::vector<int> pre, std::vector<%(float_prec)s> _weight, std::vector<int> _delay%(extra_args_vec_decl)s) {
        int i=0;

        for (int new_item = 0; new_item < pre.size(); new_item++) {
            int idx = pre_rank[post_idx].size();

            // Find where to put the synapse. As the ranks are ordered, we can
            // scan the array without begining from the start every time
            for (; i<pre_rank[post_idx].size(); i++) {
                if(pre_rank[post_idx][i] > pre[new_item]){
                    idx = i;
                    break;
                }
            }

            _add_at_position(post_idx, idx, pre[new_item], _weight[new_item], _delay[new_item]%(extra_args_vec_acc)s);
        }
    }
    void remove_single_synapse(int post_idx, int pre) {
        // Find synapse to be removed
        int pre_idx = -1;
        for(int i=0; i<pre_rank[post_idx].size(); i++){
            if(pre_rank[post_idx][i] == pre){
                pre_idx = i;
                break;
            }
        }

        // Sanity
        if (pre_idx == -1) 
            return;

        // Update connectivity
        _erase_at_position(post_idx, pre_idx);
    };

%(inverse_connectivity_rebuild)s
""",
        'pruning': """
    // Pruning
    bool _pruning;
    int _pruning_period;
    long int _pruning_offset;
""",
        'creating': """
    // Creating
    bool _creating;
    int _creating_period;
    long int _creating_offset;
""",
        'spiking_addcode': """
        // Add the corresponding pair in inv_pre_rank
        int idx_post = 0;
        for(int i=0; i<post_rank.size(); i++){
            if(post_rank[i] == post){
                idx_post = i;
                break;
            }
        }

        // we parallelize over post, therefore it could occur that two threads access the same pre-neuron
        #pragma omp critical
        {
            inv_pre_rank[pre].push_back(std::pair<int, int>(idx_post, idx));
        }
""",
        'spiking_removecode': """
        // Remove the corresponding pair in inv_pre_rank
        int pre = pre_rank[post][idx];
        for(int i=0; i<inv_pre_rank[pre].size(); i++){
            if(inv_pre_rank[pre][i].second == idx){
                // we parallelize over post, therefore it could occur that two threads access the same pre-neuron
                #pragma omp critical
                {
                    inv_pre_rank[pre].erase(inv_pre_rank[pre].begin() + i);
                    _dirty_connectivity = true;
                }
                break;
            }
        }
""",
        'spiking_rebuild_backwardview': """    bool _dirty_connectivity = false;
    void check_and_rebuild_inverse_connectivity() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "ProjStruct::check_and_rebuild_inverse_connectivity() called at time t = " << t / dt << " ms." << std::endl;
    #endif
        if (this->_dirty_connectivity) {
            inverse_connectivity_matrix();
            _dirty_connectivity = false;
        }
    }
"""
    },

    # Structural plasticity during the simulate() call
    'create': """
    // proj%(id_proj)s creating: %(eq)s
    void creating() {
        if((_creating)&&((t - _creating_offset) %% _creating_period == 0)){
        #ifdef _TRACE_SIMULATION_STEPS
            #pragma omp master
            {
            std::cout << "ProjStruct%(id_proj)s::creating() executed at time t = " << t / dt << " ms." << std::endl;
            }
        #endif
            %(proba_init)s

            #pragma omp for
            for(int i = 0; i < post_rank.size(); i++){
                int rk_post = post_rank[i];
                for(int rk_pre = 0; rk_pre < %(pre_prefix)ssize; rk_pre++){
                    if(%(condition)s){
                        // Check if the synapse exists
                        bool _exists = false;
                        for(int k=0; k<pre_rank[i].size(); k++){
                            if(pre_rank[i][k] == rk_pre){
                                _exists = true;
                                break;
                            }
                        }

                        if((!_exists)%(proba)s){
                            //std::cout << "Creating synapse between " << rk_pre << " and " << rk_post << std::endl;
                            add_single_synapse(i, rk_pre, %(weights)s%(delay)s);
                        }
                    }
                }
            }
        }
    }
""",
    'prune': """
    // proj%(id_proj)s pruning: %(eq)s
    void pruning() {
        if((_pruning)&&((t - _pruning_offset) %% _pruning_period == 0)){
        #ifdef _TRACE_SIMULATION_STEPS
            std::cout << "ProjStruct%(id_proj)s::pruning() executed at time t = " << t / dt << " ms." << std::endl;
        #endif
            %(proba_init)s

            #pragma omp for
            for(int i = 0; i < post_rank.size(); i++){
                int rk_post = post_rank[i];
                for(int j = 0; j < pre_rank[i].size();){
                    int rk_pre = pre_rank[i][j];
                    if ((%(condition)s)%(proba)s) {
                        _erase_at_position(i, j);
                    } else {
                        ++j;
                    }
                }
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
    'event_driven': event_driven,
    'rng_update': cpp_11_rng,

    # operations
    'rate_coded_sum': lil_summation_operation,
    'vectorized_default_psp': {
        'sse': {
            'single_w': lil_summation_operation_sse_single_weight,
            'multi_w': continuous_transmission_sse
        },
        'avx': {
            'single_w': lil_summation_operation_avx_single_weight,
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
    'spiking_sum_variable_delay': spiking_summation_variable_delay,
    'post_event': spiking_post_event,
    'structural_plasticity': structural_plasticity
}

conn_ids = {
    'local_index': "[i][j]",
    'semiglobal_index': '[i]',
    'global_index': '',
    'pre_index': '[pre_rank[i][j]]',
    'post_index': '[post_rank[i]]',
    'delay_nu' : '[delay[i][j]-1]', # non-uniform delay
    'delay_u' : '[delay-1]' # uniform delay
}
