"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

copy_proj_template = {
    # Declare the connectivity matrix
    'declare_connectivity_matrix': "",
    # Accessors for the connectivity matrix
    'access_connectivity_matrix': "",
    # No initiaization of the connectivity matrix
    'init_connectivity_matrix': "",
    # Export the connectivity matrix
    'export_connectivity': "",
    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': "",
    # Delays
    'wrapper_init_delay': "",
    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_copy)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_copy)s.get_pre_rank()[n]
    # Local variable w
    def get_w(self):
        return proj%(id_copy)s.get_w()
    def set_w(self, value):
        print('Cannot modify weights of a copied projection.')
    def get_dendrite_w(self, int rank):
        return proj%(id_copy)s.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, value):
        print('Cannot modify weights of a copied projection.')
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_copy)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
        print('Cannot modify weights of a copied projection.')
""",
    # Wrapper access to variables
    'wrapper_access_parameters_variables' : "",
    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;"""
}

copy_sum_template = {
    'sum': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
""",
    'max': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < pre_rank[i].size(); j++){
                if(%(psp)s > sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
""",
    'min': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < pre_rank[i].size(); j++){
                if(%(psp)s < sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
""",
    'mean': """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum/ (%(float_prec)s)(pre_rank[i].size());
        }
    }
"""
}
