#===============================================================================
#
#     LIL.py
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
    std::vector<std::vector<int>> delay;
    int max_delay;
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    max_delay = pop%(id_pre)s.max_delay;
""",
        'reset': "",
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
    },
    'nonuniform_spiking': {
        'declare': """
    std::vector<std::vector<int>> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes;
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    idx_delay = 0;
    max_delay = pop%(id_pre)s.max_delay ;
    _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay = pop%(id_pre)s.max_delay ;
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

#pragma omp for private(sum) %(schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    sum = 0.0;
    nb_pre = static_cast<%(idx_type)s>(pre_rank[i].size());
    for (%(idx_type)s j = 0; j < nb_pre; j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'max': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

#pragma omp for %(schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s j = 0;
    sum = %(psp)s ;
    for (j = 1; j < pre_rank[i].size(); j++) {
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'min': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

#pragma omp for %(schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    %(idx_type)s j = 0;
    sum = %(psp)s ;
    for (j = 1; j < pre_rank[i].size(); j++) {
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'mean': """
%(pre_copy)s
nb_post = static_cast<%(idx_type)s>(post_rank.size());

#pragma omp for %(schedule)s
for (%(idx_type)s i = 0; i < nb_post; i++) {
    sum = 0.0 ;
    for(%(idx_type)s j = 0; j < pre_rank[i].size(); j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum / static_cast<%(float_prec)s>(pre_rank[i].size());
}
"""
}

lil_summation_operation_sse_single_weight = {
    'sum' : {
        'double': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s._active) {
            %(idx_type)s _s, _stop;
            double _tmp_sum[2];
            double* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #endif
""",
        'float': """
    #ifdef __SSE4_1__
        if (_transmission && pop%(id_post)s._active) {
            %(idx_type)s _s, _stop;
            float _tmp_sum[4];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with SSE4-1 support. Please check your compiler flags ..." << std::endl;
    #endif
"""
    }
}

###############################################################
# Rate-coded continuous transmission using AVX
###############################################################
lil_summation_operation_avx_single_weight = {
    'sum' : {
        'double': """
    #ifdef __AVX__
        if (_transmission && pop%(id_post)s._active) {
            %(idx_type)s _s, _stop;
            double _tmp_sum[4];
            double* __restrict__ _pre_r = %(get_r)s;

            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
            for (%(idx_type)s i = 0; i < nb_post; i++) {
                %(idx_type)s* __restrict__ _idx = pre_rank[i].data();
                _stop = static_cast<%(idx_type)s>(pre_rank[i].size());

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

                // partial sums
                double lsum = _tmp_sum[0] + _tmp_sum[1] + _tmp_sum[2] + _tmp_sum[3];

                // remainder loop
                for (; _s < _stop; _s++)
                    lsum += _pre_r[_idx[_s]];

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __AVX__
        if (_transmission && pop%(id_post)s._active) {
            %(idx_type)s _s, _stop;
            float _tmp_sum[8];
            float* __restrict__ _pre_r = %(get_r)s;
            %(idx_type)s nb_post = static_cast<%(idx_type)s>(post_rank.size());

            #pragma omp for
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += w * lsum;
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
        if (_transmission && pop%(id_post)s._active) {
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
    #endif
    """,
        'float': """
    #ifdef __AVX__
        if (_transmission && pop%(id_post)s._active) {
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

                pop%(id_post)s._sum_%(target)s%(post_index)s += lsum;
            }
        } // active
    #else
        std::cerr << "The code was not compiled with AVX support. Please check your compiler flags ..." << std::endl;
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
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
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
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
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
spiking_summation_fixed_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    // Iterate over all incoming spikes (possibly delayed constantly)
    #pragma omp for    
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++) {
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
            #pragma omp critical
            {
                %(g_target)s
            }

            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""

spiking_summation_fixed_delay_inner_loop = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    // Iterate over all incoming spikes (possibly delayed constantly)
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++) {
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
if (_transmission && pop%(id_post)s._active){

    // Iterate over the spikes emitted during the last step in the pre population
    #pragma omp for
    for(int idx_spike=0; idx_spike<pop%(id_pre)s.spiked.size(); idx_spike++){

        // Get the rank of the pre-synaptic neuron which spiked
        int rk_pre = pop%(id_pre)s.spiked[idx_spike];
        // List of post neurons receiving connections
        std::vector< std::pair<int, int> > rks_post = inv_pre_rank[rk_pre];

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
if(_transmission && pop%(id_post)s._active){

    #pragma omp for
    for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
        // In which sub matrix the neuron take place
        int rk_post = pop%(id_post)s.spiked[_idx_i];

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
    // Structural plasticity
    int dendrite_index(int post, int pre){
        int idx = -1;
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] == pre){
                idx = i;
                break;
            }
        }
        return idx;
    }
    void addSynapse(int post, int pre, double weight, int _delay=0%(extra_args)s){
        // Find where to put the synapse
        int idx = pre_rank[post].size();
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] > pre){
                idx = i;
                break;
            }
        }

        // Update connectivty
        pre_rank[post].insert(pre_rank[post].begin() + idx, pre);
        w[post].insert(w[post].begin() + idx, weight);

        // Update additional fields
%(delay_code)s
%(add_code)s
%(spike_add)s
%(rd_add)s
    };
    void removeSynapse(int post, int idx){
        pre_rank[post].erase(pre_rank[post].begin() + idx);
        w[post].erase(w[post].begin() + idx);
%(delay_remove)s
%(add_remove)s
%(spike_remove)s
%(rd_remove)s
    };
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
        inv_pre_rank[pre].push_back(std::pair<int, int>(idx_post, idx));
""",
        'spiking_removecode': """
        // Remove the corresponding pair in inv_pre_rank
        int pre = pre_rank[post][idx];
        for(int i=0; i<inv_pre_rank[pre].size(); i++){
            if(inv_pre_rank[pre][i].second == idx){
                inv_pre_rank[pre].erase(inv_pre_rank[pre].begin() + i);
                break;
            }
        }
"""
    },
    'pyx_struct': {
        'pruning':
"""
        # Pruning
        bool _pruning
        int _pruning_period
        long _pruning_offset
""",
        'creating':
"""
        # Creating
        bool _creating
        int _creating_period
        long _creating_offset
""",
        'func':
"""
        # Structural plasticity
        int dendrite_index(int post, int pre)
        void addSynapse(int post, int pre, double weight, int _delay%(extra_args)s)
        void removeSynapse(int post, int pre)
"""
    },
    'pyx_wrapper': {
        'pruning':
"""
    # Pruning
    def start_pruning(self, int period, long offset):
        proj%(id)s._pruning = True
        proj%(id)s._pruning_period = period
        proj%(id)s._pruning_offset = offset
    def stop_pruning(self):
        proj%(id)s._pruning = False
""",
        'creating':
"""
    # Creating
    def start_creating(self, int period, long offset):
        proj%(id)s._creating = True
        proj%(id)s._creating_period = period
        proj%(id)s._creating_offset = offset
    def stop_creating(self):
        proj%(id)s._creating = False
""",
        'func':
"""
    # Structural plasticity
    def dendrite_index(self, int post_rank, int pre_rank):
        return proj%(id)s.dendrite_index(post_rank, pre_rank)
    def add_synapse(self, int post_rank, int pre_rank, double weight, int delay%(extra_args)s):
        proj%(id)s.addSynapse(post_rank, pre_rank, weight, delay%(extra_values)s)
    def remove_synapse(self, int post_rank, int pre_rank):
        proj%(id)s.removeSynapse(post_rank, proj%(id)s.dendrite_index(post_rank, pre_rank))
"""
    }
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
            'single_w': lil_summation_operation_sse_single_weight
        },
        'avx': {
            'multi_w': lil_summation_operation_avx,
            'single_w': lil_summation_operation_avx_single_weight
        }
    },
    'update_variables': update_variables,
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay_inner_loop,
    'spiking_sum_variable_delay': spiking_summation_variable_delay,
    'post_event': spiking_post_event,
    'structural_plasticity': structural_plasticity
}
