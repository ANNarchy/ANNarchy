pooling_template = {
    'include_additional': '#include <limits>',

    # Declare the connectivity matrix
    'declare_connectivity_matrix': """
    std::vector<int> post_rank;
    std::vector< std::vector<int> > pre_rank;
    """,

    # Accessors for the connectivity matrix
    'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }
""",

    # Export the connectivity matrix
    'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
""",

    # Arguments to the wrapper constructor
    'wrapper_args': "weights, coords",

    # Initialize the wrapper connectivity matrix
    'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_rank(coords)
""",
    # Wrapper access to connectivity matrix
    'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()
            """,

    # Wrapper access to variables
    'wrapper_access_parameters_variables': "",

    # Variables for the psp code
    'psp_prefix': """
        int rk_pre;
        double sum=%(sum_default)s;
    """,

    # Override the monitor to avoid recording the weights
    'monitor_class':"""
class ProjRecorder%(id_proj)s : public Monitor
{
public:
    ProjRecorder%(id_proj)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
    };
    void record() {
    };
    void record_targets() { /* nothing to do here */ }

};
""",
    'monitor_export': """
    # Projection %(id_proj)s : Monitor
    cdef cppclass ProjRecorder%(id_proj)s (Monitor):
        ProjRecorder%(id_proj)s(vector[int], int, long) except +
""",
     'monitor_wrapper': """
# Projection %(id_proj)s: Monitor wrapper
cdef class ProjRecorder%(id_proj)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, long offset):
        self.thisptr = new ProjRecorder%(id_proj)s(ranks, period, offset)
"""
}
