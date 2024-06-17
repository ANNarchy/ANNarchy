"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

omp_population = {
    'template': """
class PopRecorder%(id)s : public Monitor
{
protected:
    PopRecorder%(id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s (" << this << ") instantiated." << std::endl;
    #endif
%(init_code)s
    }

public:
    ~PopRecorder%(id)s() {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s::~PopRecorder%(id)s() - this = " << this << std::endl;
    #endif
    }

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder%(id)s(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder%(id)s* get_instance(int id) {
        return static_cast<PopRecorder%(id)s*>(getRecorder(id));
    }

    void record() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder%(id)s::record()" << std::endl;
    #endif
%(recording_code)s
    }

    void record_targets() {
%(recording_target_code)s
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(size_in_bytes)s
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s::clear() - this = " << this << std::endl;
    #endif
%(clear_monitor_code)s

        removeRecorder(this);
    }


%(struct_code)s
};
""",
    'local': {
    'struct': """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ; """,
    'init': """
        this->%(name)s = std::vector< std::vector< %(type)s > >();
        this->record_%(name)s = false; """,
    'recording': """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->%(name)s.push_back(pop%(id)s.%(name)s);
            else{
                std::vector<%(type)s> tmp = std::vector<%(type)s>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop%(id)s.%(name)s[this->ranks[i]]);
                }
                this->%(name)s.push_back(tmp);
            }
        }""",
    'size_in_bytes': """
// local variable %(name)s
size_in_bytes += sizeof(std::vector<%(type)s>) * %(name)s.capacity();
for(auto it=%(name)s.begin(); it!= %(name)s.end(); it++) {
    size_in_bytes += it->capacity() * sizeof(%(type)s);
}""",
    'clear': """
        for(auto it = this->%(name)s.begin(); it != this->%(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->%(name)s.clear();
    """
    },
    'semiglobal': { # Does not exist for populations
        'struct': "",
        'init': "",
        'recording': "",
        'size_in_bytes': "",
        'clear': ""
    },
    'global': {
        'struct': """
    // Global variable %(name)s
    std::vector< %(type)s > %(name)s ;
    bool record_%(name)s ; """,
        'init': """
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false; """,
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            this->%(name)s.push_back(pop%(id)s.%(name)s);
        } """,
        'size_in_bytes': """
// global variable %(name)s
size_in_bytes += sizeof(%(type)s);""",
        'clear': """
        this->%(name)s.clear();
    """
    }
}

omp_projection = {
    'struct': """
class ProjRecorder%(id)s : public Monitor
{
protected:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder%(id)s (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj%(id)s.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();
%(init_code)s

    };

    std::vector <int> indices;

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder%(id)s(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder%(id)s (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder%(id)s* get_instance(int id) {
        return static_cast<ProjRecorder%(id)s*>(getRecorder(id));
    }

    void record() {
%(recording_code)s
    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        size_t size_in_bytes = 0;

%(size_in_bytes_code)s

        return static_cast<long int>(size_in_bytes);
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor%(id)s::clear()." << std::endl;
    #endif
%(clear_code)s
    }

%(struct_code)s
};
""",
    'local': {
        'struct': """
    // Local variable %(name)s
    std::vector< std::vector< std::vector< %(type)s > > > %(name)s ;
    bool record_%(name)s ;
""",
        'init' : """
        this->%(name)s = std::vector< std::vector< std::vector< %(type)s > > >();
        this->record_%(name)s = false;
""",
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            std::vector< std::vector< %(type)s > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj%(id)s.get_matrix_variable_row<%(type)s>(proj%(id)s.%(name)s, this->indices[i])));
            }
            this->%(name)s.push_back(tmp);
            tmp.clear();
        }
""",
        'clear': """
for (auto it=%(name)s.begin(); it!= %(name)s.end(); it++) {
    for (auto it2=it->begin(); it2!= it->end(); it2++) {
        it2->clear();
        it2->shrink_to_fit();
    }
}
""",
        'size_in_bytes': """
// local variable %(name)s
size_in_bytes += sizeof(std::vector<std::vector<%(type)s>>) * %(name)s.capacity();
for (auto it=%(name)s.begin(); it!= %(name)s.end(); it++)
    for (auto it2=it->begin(); it2!= it->end(); it2++)
        size_in_bytes += it2->capacity() * sizeof(%(type)s);
"""
    },
    'semiglobal': {
        'struct': """
    // Semiglobal variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ;
""",
        'init' : """
        this->%(name)s = std::vector< std::vector< %(type)s > >();
        this->record_%(name)s = false;
""",
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            std::vector< %(type)s > tmp;
            auto value = std::move(proj%(id)s.get_vector_variable_all<%(type)s>(proj%(id)s.%(name)s));
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(value[this->indices[i]]);
            }
            this->%(name)s.push_back(tmp);
            tmp.clear();
        }
""",
        'clear': """
// semiglobal variable %(name)s
for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
    it->clear();
    it->shrink_to_fit();
}
%(name)s.clear();
%(name)s.shrink_to_fit();
""",
        'size_in_bytes': """
// semiglobal variable %(name)s
size_in_bytes += sizeof(std::vector<%(type)s>) * %(name)s.capacity();
for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
    size_in_bytes += sizeof(%(type)s) * it->capacity();
}
"""
    },
    'global': {
        'struct': """
    // Global variable %(name)s
    std::vector< %(type)s > %(name)s ;
    bool record_%(name)s ;
""",
        'init' : """
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false;
""",
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            this->%(name)s.push_back(proj%(id)s.%(name)s);
        }
""",
        'clear': """
// global variable %(name)s
%(name)s.clear();
%(name)s.shrink_to_fit();
""",
        'size_in_bytes': """
// global variable %(name)s
size_in_bytes += sizeof(%(type)s) * %(name)s.capacity();
"""
    }
}