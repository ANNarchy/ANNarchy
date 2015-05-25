header_struct_template = """#pragma once

#include "%(pre_name)s.hpp"
#include "%(post_name)s.hpp"

extern PopStruct%(pre_id)s pop%(pre_id)s;
extern PopStruct%(post_id)s pop%(post_id)s;

/////////////////////////////////////////
// proj%(id_proj)s: %(pre_name)s -> %(post_name)s with target %(target)s
/////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // number of dendrites
    int size;

    // Learning flag
    bool _learning;

    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

%(code)s

    void init_projection() {
%(init)s
    }

    void compute_psp() {
%(psp_prefix)s

%(psp)s
    }
    
    void update_synapse() {
        int rk_pre, rk_post;

%(update)s
    }
};
"""