connectivity_matrix_omp = {
    'declare': """
    int _num_rows;
    int _num_columns;
""",
    'accessor': """
    // Accessor to connectivity data
    int num_rows() { return _num_rows; }
    int num_columns() { return _num_columns; }
    void set_mat_dims(int num_rows, int num_columns) { 
        _num_columns = num_columns;
        _num_rows = num_rows;

        // local variables
%(variable)s
    }
    int nb_synapses(int n) { return _num_columns; }
""",
    'init': """
        _num_rows = pop%(id_post)s.size;
        _num_columns = pop%(id_pre)s.size;    
""",
    'pyx_struct': """
        # Dense Connectivity
        int num_rows()
        int num_columns()
        void set_mat_dims(int, int)
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef CSR syn = synapses
        cdef int size = syn.size
        proj%(id_proj)s.set_size( size )
        proj%(id_proj)s.set_mat_dims( %(post_size)s, %(pre_size)s )
""",
    'pyx_wrapper_accessor': """
    # Connectivity
    def post_rank(self):
        return range(proj%(id_proj)s.num_rows())
    def pre_rank(self, int n):
        return range(proj%(id_proj)s.num_columns())
    def pre_rank_all(self):
        tmp_row = np.arange(proj%(id_proj)s.num_columns())
        pre_ranks = []
        for i in range(proj%(id_proj)s.num_rows()):
            pre_ranks.append(tmp_row)
        return pre_ranks
"""
}

# Inverse connectivity is simply index switch, so
# nothing to implement here
inverse_connectivity_matrix = {
    'declare': "",
    'init': ""
}

weight_matrix_omp = {
    'declare': """
    // dense weights
    std::vector<double> w;
""",
    'accessor': """
    // Local parameter w
    std::vector< double > get_w() { return w; }
    std::vector<double> get_dendrite_w(int rk) { return std::vector<double>(w.begin()+rk*_num_columns, w.begin()+(rk+1)*_num_columns); }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post*_num_rows+rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) {
        // set matrix data row-by-row
        for(auto i=0; i < value.size(); i++)
            std::copy(value[i].begin(), value[i].end(), w.begin()+i*_num_columns);
    }
    void set_dendrite_w(int rk, std::vector<double> value) {
        std::copy(value.begin(), value.end(), w.begin()+rk*_num_columns);
    }
    void set_synapse_w(int rk_post, int rk_pre, double value) { *(w.begin()+rk_post*_num_columns+rk_pre) = value; }
""",
    'init': """
""",
    'pyx_struct': """
        # Local variable w
        vector[double] get_w()
        vector[double] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[double]])
        void set_dendrite_w(int, vector[double])
        void set_synapse_w(int, int, double)
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w(syn.w)
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        data = proj%(id_proj)s.get_w()
        return np.array(data).reshape(proj%(id_proj)s.num_rows(), proj%(id_proj)s.num_columns())
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

single_weight_matrix_omp = {
    'declare': """
    // Single weight in the projection
    double w;
""",
    'accessor': "",
    'init': "",
    'pyx_struct': """
        # Local variable w
        double w
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        # Use only the first weight
        proj%(id_proj)s.w = syn.w[0][0]
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.w
    def set_dendrite_w(self, int rank, double value):
        proj%(id_proj)s.w = value
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.w
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.w = value
"""
}

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
"""
}

attribute_acc = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    std::vector< %(type)s > get_dendrite_%(name)s(int rk) { return std::vector< %(type)s > (%(name)s.begin()+rk*_num_columns, %(name)s.begin()+(rk+1)*_num_columns); }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return %(name)s[rk_post*_num_columns+rk_pre]; }
    void set_%(name)s(std::vector< %(type)s >value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { std::copy(value.begin(), value.end(), %(name)s.begin()+ rk*_num_columns); }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { %(name)s[rk_post*_num_columns+rk_pre] = value; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector< %(type)s >( _num_rows * _num_columns, %(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>( _num_rows, %(init)s);
"""
}

delay = {
    'declare': """
    // Non-uniform delay
    std::vector< int > delay ;""",
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

conn_templates = {
    # connectivity
    'connectivity_matrix': connectivity_matrix_omp,
    'inverse_connectivity_matrix': inverse_connectivity_matrix,
    'weight_matrix': weight_matrix_omp,
    'single_weight_matrix': single_weight_matrix_omp,
    
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_acc':attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'delay': delay,
    'event_driven': None
}