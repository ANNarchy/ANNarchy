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
        if(%(name)s.empty())
            return false;
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
        // Global %(attr_type)s %(name)s
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
"""
}

######################################
### Dense Matrix templates
######################################
dense_summation_operation = {
    'sum' : """
%(pre_copy)s

// matrix dimensions
%(idx_type)s rows = %(post_prefix)ssize;
%(idx_type)s columns = %(pre_prefix)ssize;

// running indices
%(idx_type)s i;
%(size_type)s j;
%(idx_type)s rk_pre;

for(i = 0; i < rows; i++) {
    sum = 0.0;
    rk_pre = 0;
    j=i*columns;

    for ( ; rk_pre < columns; j++, rk_pre++) {
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s[i] += sum;
}
"""
}

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using SIMD
# intrinsics
###############################################################################
continuous_transmission_sse = {
    'sum': {
        'double': """
#ifdef __SSE4_1__
    if (_transmission && pop%(id_post)s._active) {
        double _tmp_sum[2];

        // matrix dimensions
        %(idx_type)s rows = pop%(id_post)s.size;
        %(idx_type)s columns = pop%(id_pre)s.size;

        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        double* __restrict__ _pre_r = %(get_r)s;
        double* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            %(idx_type)s rk_post = i;
            __m128d _tmp_reg_sum = _mm_setzero_pd();

            _s=i*columns;
            for (j = 0; (j+8) < columns; j+=8, _s+=8) {
                __m128d _tmp_r = _mm_loadu_pd(&_pre_r[j]);
                __m128d _tmp_r2 = _mm_loadu_pd(&_pre_r[j+2]);
                __m128d _tmp_r3 = _mm_loadu_pd(&_pre_r[j+4]);
                __m128d _tmp_r4 = _mm_loadu_pd(&_pre_r[j+6]);

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
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
#endif
""",
        'float': """
#ifdef __SSE4_1__
    if (_transmission && pop%(id_post)s._active) {
        float _tmp_sum[4];

        // matrix dimensions
        %(idx_type)s rows = pop%(id_post)s.size;
        %(idx_type)s columns = pop%(id_pre)s.size;

        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        float* __restrict__ _pre_r = %(get_r)s;
        float* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            %(idx_type)s rk_post = i;
            __m128 _tmp_reg_sum = _mm128_setzero_ps();

            _s=i*columns;
            for (j = 0; (j+16) < columns; j+=16, _s+=16) {
                __m128 _tmp_r = _mm_loadu_ps(&_pre_r[j]);
                __m128 _tmp_r2 = _mm_loadu_ps(&_pre_r[j+2]);
                __m128 _tmp_r3 = _mm_loadu_ps(&_pre_r[j+4]);
                __m128 _tmp_r4 = _mm_loadu_ps(&_pre_r[j+6]);

                __m128 _tmp_w = _mm_loadu_ps(&_w[_s]);
                __m128 _tmp_w2 = _mm_loadu_ps(&_w[_s+2]);
                __m128 _tmp_w4 = _mm_loadu_ps(&_w[_s+4]);
                __m128 _tmp_w6 = _mm_loadu_ps(&_w[_s]+6);

                _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r, _tmp_w));
                _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r2, _tmp_w2));
                _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r3, _tmp_w3));
                _tmp_reg_sum = _mm_add_ps(_tmp_reg_sum, _mm_mul_ps(_tmp_r4, _tmp_w4s));
            }

            _mm256_storeu_ps(_tmp_sum, _tmp_reg_sum);

            // partial sums
            float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

            // remainder loop
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
#endif
"""
    }
}

continuous_transmission_avx = {
    'sum': {
        'double': """
#ifdef __AVX__
    if (_transmission && %(post_prefix)s_active) {
        double _tmp_sum[4];

        // matrix dimensions
        %(idx_type)s rows = %(post_prefix)ssize;
        %(idx_type)s columns = %(pre_prefix)ssize;
        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        double* __restrict__ _pre_r = %(get_r)s;
        double* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            __m256d _tmp_reg_sum = _mm256_setzero_pd();

            _s=i*columns;
            for (j = 0; (j+8) < columns; j+=8, _s+=8) {
                __m256d _tmp_r = _mm256_loadu_pd(&_pre_r[j]);
                __m256d _tmp_r2 = _mm256_loadu_pd(&_pre_r[j+4]);

                __m256d _tmp_w = _mm256_loadu_pd(&_w[_s]);
                __m256d _tmp_w2 = _mm256_loadu_pd(&_w[_s+4]);

                _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _mm256_mul_pd(_tmp_r, _tmp_w));
                _tmp_reg_sum = _mm256_add_pd(_tmp_reg_sum, _mm256_mul_pd(_tmp_r2, _tmp_w2));
            }

            _mm256_storeu_pd(_tmp_sum, _tmp_reg_sum);

            // partial sums
            double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

            // remainder loop
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            %(post_prefix)s_sum_%(target)s[i] += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
#endif
""",
    'float': """
    #ifdef __AVX__
    if (_transmission && %(post_prefix)s_active) {
        float _tmp_sum[8];

        // matrix dimensions
        %(idx_type)s rows = %(post_prefix)ssize;
        %(idx_type)s columns = %(pre_prefix)ssize;
        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        float* __restrict__ _pre_r = %(get_r)s;
        float* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            __m256 _tmp_reg_sum = _mm256_setzero_ps();

            _s=i*columns;
            for (j = 0; (j+16) < columns; j+=16, _s+=16) {
                __m256 _tmp_r = _mm256_loadu_ps(&_pre_r[j]);
                __m256 _tmp_r2 = _mm256_loadu_ps(&_pre_r[j+8]);

                __m256 _tmp_w = _mm256_loadu_ps(&_w[_s]);
                __m256 _tmp_w2 = _mm256_loadu_ps(&_w[_s+8]);

                _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _mm256_mul_ps(_tmp_r, _tmp_w));
                _tmp_reg_sum = _mm256_add_ps(_tmp_reg_sum, _mm256_mul_ps(_tmp_r2, _tmp_w2));
            }

            _mm256_storeu_ps(_tmp_sum, _tmp_reg_sum);

            // partial sums
            float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

            // remainder loop
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            %(post_prefix)s_sum_%(target)s[i] += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
#endif
"""
    }
}

continuous_transmission_avx512 = {
    'sum': {
        'double': """
#ifdef __AVX512F__
    if (_transmission && pop%(id_post)s._active) {
        double _tmp_sum[8];

        // matrix dimensions
        %(idx_type)s rows = pop%(id_post)s.size;
        %(idx_type)s columns = pop%(id_pre)s.size;

        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        double* __restrict__ _pre_r = %(get_r)s;
        double* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            %(idx_type)s rk_post = i;
            __m512d _tmp_reg_sum = _mm512_setzero_pd();

            _s=i*columns;
            for (j = 0; (j+8) < columns; j+=8, _s+=8) {
                __m512d _tmp_r = _mm512_loadu_pd(&_pre_r[j]);
                __m512d _tmp_w = _mm512_loadu_pd(&_w[_s]);

                _tmp_reg_sum = _mm512_add_pd(_tmp_reg_sum, _mm512_mul_pd(_tmp_r, _tmp_w));
            }

            _mm512_storeu_pd(_tmp_sum, _tmp_reg_sum);

            // partial sums
            double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

            // remainder loop
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
#endif
""",
        'float': """
#ifdef __AVX512F__
    if (_transmission && pop%(id_post)s._active) {
        float _tmp_sum[16];

        // matrix dimensions
        %(idx_type)s rows = pop%(id_post)s.size;
        %(idx_type)s columns = pop%(id_pre)s.size;

        // running indices
        %(idx_type)s i, j;
        %(size_type)s _s;

        // required pointer
        float* __restrict__ _pre_r = %(get_r)s;
        float* __restrict__ _w = w.data();

        // Row-wise SpMV
        for(i = 0; i < rows; i++) {
            %(idx_type)s rk_post = i;
            __m512 _tmp_reg_sum = _mm512_setzero_ps();

            _s=i*columns;
            for (j = 0; (j+16) < columns; j+=16, _s+=16) {
                __m512 _tmp_r = _mm512_loadu_ps(&_pre_r[j]);
                __m512 _tmp_w = _mm512_loadu_ps(&_w[_s]);

                _tmp_reg_sum = _mm512_add_ps(_tmp_reg_sum, _mm512_mul_ps(_tmp_r, _tmp_w));
            }

            _mm512_storeu_ps(_tmp_sum, _tmp_reg_sum);

            // partial sums
            float lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7] + _tmp_sum[8] + _tmp_sum[9] + _tmp_sum[10] + _tmp_sum[11] + _tmp_sum[12] + _tmp_sum[13] + _tmp_sum[14] + _tmp_sum[15];

            // remainder loop
            for (; j < columns; j++, _s++)
                lsum += _pre_r[j] * _w[_s];

            pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
        }
    } // active
#else
    std::cerr << "The code was not compiled with AVX-512 support. Please check your compiler flags ..." << std::endl;
#endif
"""
    }
}

# HD (19th May 2022):
# Our default strategy, to loop over all spike events and update post.g_target can not applied here
# as it would lead to 100% cache misses and an enormously high number of memory stalls.
spiking_summation_fixed_delay = """// Event-based summation
if (_transmission && %(post_prefix)s_active){

    for (%(idx_type)s rk_post = 0; rk_post < num_rows(); rk_post++) {
        // Iterate over all spiking neurons
        for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
            %(idx_type)s rk_pre = *it;
            %(size_type)s j = rk_post*this->num_columns_ + rk_pre;

            if (mask_[j]) {
                %(event_driven)s
                %(g_target)s
                %(pre_event)s
            }
        }
    }
} // active
"""

dense_update_variables = {
    'local': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s

    // Local variables
    for(%(idx_type)s i = 0; i < %(post_prefix)ssize; i++){
        rk_post = i; // dense: ranks are indices

        // Semi-global variables
    %(semiglobal)s

        // Local variables are updated to boolean flag
        %(size_type)s j = i*%(pre_prefix)ssize;
        for(rk_pre = 0; rk_pre < %(pre_prefix)ssize; rk_pre++, j++) {
            if(mask_[j]) {
%(local)s
            }
        }
    }
}
""",
    'global': """
// Check periodicity
if(_transmission && _update && %(post_prefix)s_active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s

    // Semi-global variables
    for(int i = 0; i < %(post_prefix)ssize; i++){
        rk_post = i;
    %(semiglobal)s
    }
}
"""
}

spiking_post_event = """
if (_transmission && %(post_prefix)s_active) {

    %(idx_type)s columns = pop%(id_pre)s.size;

    for (%(idx_type)s _idx_i = 0; _idx_i < %(post_prefix)sspiked.size(); _idx_i++) {
        %(idx_type)s post_rank = %(post_prefix)sspiked[_idx_i];
        %(idx_type)s rk_pre = 0;

        for (%(size_type)s j = post_rank * columns; j < (post_rank+1) * columns; j++, rk_pre++) {
            if (mask_[j]) {
%(event_driven)s
%(post_event)s
            }
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

    #operations
    'rate_coded_sum': dense_summation_operation,
    'vectorized_default_psp': {
        'sse': {
            'multi_w': continuous_transmission_sse
        },
        'avx': {
            'multi_w': continuous_transmission_avx
        },
        'avx512': {
            'multi_w': continuous_transmission_avx512
        }
    },
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'update_variables': dense_update_variables,
    'post_event': spiking_post_event,
    'event_driven': event_driven
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[rk_post]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}
