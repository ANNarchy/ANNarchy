# Definition of a population as a c-like struct, divided
# into two groups: rate or spike
# 
# Parameters:
#    id: id of the population
#    additional: neuron specific definitions
header_struct = {
    'rate' :
"""
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

%(additional)s

    // Record parameter
    int record_period;
    long int record_offset;
    std::vector<int> record_ranks;

    // Access functions used by cython wrapper
int get_size() { return size; }
bool is_active() { return _active; }
bool set_active(bool val) { _active = val; }

    // Record
    void set_record_period(int period, long int t) { record_period = period; record_offset = t; }
    void set_record_ranks( std::vector<int> ranks) { record_ranks = ranks; }

    // Neuron specific
%(accessor)s
};
""",
    'spike':
"""
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

%(additional)s

    // Spiking events
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;

    // Record parameter
    int record_period;
    long int record_offset;
    std::vector<int> record_ranks;

    // Access functions used by cython wrapper
int get_size() { return size; }
bool is_active() { return _active; }
bool set_active(bool val) { _active = val; }

    // Record
    void set_record_period(int period, long int t) { record_period = period; record_offset = t; }
    void set_record_ranks( std::vector<int> ranks) { record_ranks = ranks; }

    // Neuron specific
%(accessor)s
};
"""
}

parameter_decl = {
    'local':
"""
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'global':
"""
    // Global parameter %(name)s
    %(type)s  %(name)s ;
"""    
}

parameter_acc = {
    'local':
"""
    // Local parameter %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; }
""",
    'global':
"""
    // Global parameter %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s val) { %(name)s = val; }
"""
}

variable_decl = {
    'local':
"""
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""",
    'global':
"""
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
"""
}

parameter_cpp_export = {
    'local':
"""
        # Local parameter %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_single_%(name)s(int rk)
        void set_%(name)s(vector[%(type)s])
        void set_single_%(name)s(int, %(type)s)
""",
    'global':
"""
        # Global parameter %(name)s
        %(type)s  get_%(name)s()
        void set_%(name)s(%(type)s)
"""
}

parameter_pyx_wrapper = {
    'local':
"""
    # Local parameter %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.get_%(name)s())
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.set_%(name)s( value )
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.get_single_%(name)s(rank)
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.set_single_%(name)s(rank, value)
""",
    'global':
"""
    # Global parameter %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.get_%(name)s()
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.set_%(name)s(value)
"""
}

variable_acc = {
    'local':
"""
    // Local variable %(name)s
    std::vector< %(type)s > get_%(name)s() { return %(name)s; }
    %(type)s get_single_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector< %(type)s > val) { %(name)s = val; }
    void set_single_%(name)s(int rk, %(type)s val) { %(name)s[rk] = val; }
    std::vector< std::vector< %(type)s > > get_recorded_%(name)s() { return recorded_%(name)s; }
    bool is_%(name)s_recorded() { return record_%(name)s; }
    void set_record_%(name)s(bool val) { record_%(name)s = val; }
    void clear_recorded_%(name)s() { recorded_%(name)s.clear(); }
""",
    'global':
"""
    // Global variable %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s val) { %(name)s = val; }
    std::vector<%(type)s> get_recorded_%(name)s() { return recorded_%(name)s; }
    bool is_%(name)s_recorded() { return record_%(name)s; }
    void set_record_%(name)s(bool val) { record_%(name)s = val; }
    void clear_recorded_%(name)s() { recorded_%(name)s.clear(); }
"""
}

variable_cpp_export = {
    'local':
"""
        # Local variable %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_single_%(name)s(int rk)
        void set_%(name)s(vector[%(type)s])
        void set_single_%(name)s(int, %(type)s)
        vector[vector[%(type)s]] get_recorded_%(name)s()
        void set_record_%(name)s(bool)
        bool is_%(name)s_recorded()
        void clear_recorded_%(name)s()
""",
    'global':
"""
        # Global variable %(name)s
        %(type)s  get_%(name)s()
        void set_%(name)s(%(type)s)
        vector[%(type)s] get_recorded_%(name)s()
        void set_record_%(name)s(bool)
        bool is_%(name)s_recorded()
        void clear_recorded_%(name)s()
"""
}

variable_pyx_wrapper = {
    'local':
"""
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.get_%(name)s())
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.set_%(name)s(value)
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.get_single_%(name)s(rank)
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.set_single_%(name)s(rank, value)
    def start_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(True)
    def stop_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(False)
    def get_record_%(name)s(self):
        cdef vector[vector[%(type)s]] tmp = pop%(id)s.get_recorded_%(name)s()
        pop%(id)s.clear_recorded_%(name)s()
        return tmp
""",
    'global':
"""
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.get_%(name)s()
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.set_%(name)s(value)
    def start_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(True)
    def stop_record_%(name)s(self):
        pop%(id)s.set_record_%(name)s(False)
    def get_record_%(name)s(self):
        cdef vector[%(type)s] tmp = pop%(id)s.get_recorded_%(name)s()
        pop%(id)s.clear_recorded_%(name)s()
        return tmp
"""
}