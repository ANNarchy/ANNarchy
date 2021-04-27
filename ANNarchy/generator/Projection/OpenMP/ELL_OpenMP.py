#===============================================================================
#
#     ELL_OpenMP.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

###############################################################
# Rate-coded continuous transmission
###############################################################
ell_summation_operation = {
    'sum' : """
%(pre_copy)s

%(float_prec)s* __restrict__ target = pop%(id_post)s._sum_%(target)s.data();

#pragma omp for firstprivate(maxnzr_)
for(int i = 0; i < post_ranks_.size(); i++) {
    rk_post = post_ranks_[i]; // Get postsynaptic rank

    sum = 0.0;
    for(int j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++) {
        rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }
    
    target%(post_index)s += sum;
}""",
    'max': "",
    'min': "",
    'mean': "",
}

ell_summation_operation_simd = {
    'sum' : """
%(pre_copy)s

%(float_prec)s* __restrict__ target = pop%(id_post)s._sum_%(target)s.data();

#pragma omp for firstprivate(maxnzr_)
for(int i = 0; i < post_ranks_.size(); i++) {
    rk_post = post_ranks_[i]; // Get postsynaptic rank
    sum = 0.0;

    double pre_r[%(simd_len)s];
    int j = i*maxnzr_;

    // SIMD partition
    for(; j + %(simd_len)s < i*maxnzr_+rl_[i]; j += %(simd_len)s) {
        int* __restrict__ col_idx_ptr = &col_idx_[j];
        double *__restrict__ w_ptr = &w[j];

        // pre-load uncoalesced values
        for(int j2 = 0; j2 < %(simd_len)s; j2++)
            pre_r[j2] = pop%(id_pre)s.r[col_idx_ptr[j2]];

        // sum up
        #pragma omp simd
        for(int j2 = 0; j2 < %(simd_len)s; j2++)
            sum += w_ptr[j2] * pre_r[j2];
    }

    // remainder loop
    for(; j < i*maxnzr_+rl_[i]; j++) {
        rk_pre = col_idx_[j];
        sum += %(psp)s ;
    }

    target%(post_index)s += sum;
}""",
    'max': "",
    'min': "",
    'mean': "",
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
    for(int i = 0; i < post_ranks_.size(); i++){
        rk_post = post_ranks_[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s
        // Local variables
        for(int j = i*maxnzr_; j < i*maxnzr_+rl_[i]; j++){
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
    
    'rate_coded_sum': ell_summation_operation,
    'update_variables': update_variables
}