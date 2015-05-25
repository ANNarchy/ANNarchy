# Definition of a population as a c-like struct, divided
# into two groups: rate or spike
# 
# Parameters:
#    id: id of the population
#    additional: neuron specific definitions
header_struct = {
    'rate' : \
"""
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Active
    bool _active;

    std::vector<double> get_r() {
        return r;
    }

    double get_single_r(int rank) {
        return r[rank];
    }
    
%(additional)s

    // Record parameter
    int record_period;
    long int record_offset;
    std::vector<int> record_ranks;
};
""",
    'spike' : \
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
};
"""
}

parameter_decl = {
    'local': \
"""
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
""",
    'global': \
"""
    // Global parameter %(name)s
    %(type)s  %(name)s ;
"""    
}

variable_decl = {
    'local': \
"""
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""",
    'global': \
"""
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
"""
}