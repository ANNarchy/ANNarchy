#===============================================================================
#
#     Dense.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Julien Vitay <julien.vitay@gmail.com>
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
%(idx_type)s i, j;

for(i = 0; i < rows; i++) {
    sum = 0.0;
    for(%(idx_type)s rk_pre = 0, j=i*columns; rk_pre < columns; j++, rk_pre++) {
        sum += %(psp)s ;
    }
    %(post_prefix)s_sum_%(target)s[i] += sum;
}
"""
}

###############################################################################
# Optimized kernel for default rate-coded continuous transmission using AVX
#
# For details on single_weight: see lil_summation_operation_sse_single_weight
###############################################################################
dense_summation_operation_avx = {
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
            double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3] + _tmp_sum[4] + _tmp_sum[5] + _tmp_sum[6] + _tmp_sum[7];

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

spiking_summation_fixed_delay_csr = """// Event-based summation
if (_transmission && %(post_prefix)s_active){

    // Iterate over all spiking neurons
    for (auto it = %(pre_prefix)sspiked.cbegin(); it != %(pre_prefix)sspiked.cend(); it++) {
        %(size_type)s beg = (*it) * this->num_columns_;
        %(size_type)s end = (*it+1) * this->num_columns_;

        %(idx_type)s rk_post = 0;

        // Iterate over columns
        for (%(idx_type)s j = beg; j < end; j++, rk_post++) {
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
        'avx': {
            'multi_w': dense_summation_operation_avx
        },
    },
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay_csr,    
    'update_variables': dense_update_variables,
}

conn_ids = {
    'local_index': '[j]',
    'semiglobal_index': '[rk_post]',
    'global_index': '',
    'post_index': '[rk_post]',
    'pre_index': '[rk_pre]',
    'delay_u' : '[delay-1]' # uniform delay
}