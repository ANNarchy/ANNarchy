"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

cuda_population = {
    'template': """
class PopRecorder%(id)s : public Monitor
{
protected:
    int _id;

public:
    PopRecorder%(id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
        {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s (" << this << ") instantiated." << std::endl;
    #endif
%(init_code)s

        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif
    }

    ~PopRecorder%(id)s() {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s::~PopRecorder%(id)s - this = " << this << std::endl;
    #endif
    }

    void record() {
%(recording_code)s
    }

    void record_targets() {
%(recording_target_code)s
    }

    long int size_in_bytes() {
        size_t size_in_bytes = 0;
%(size_in_bytes)s
        return static_cast<long int>(size_in_bytes);
    }

%(clear_individual_container_code)s

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopRecorder%(id)s::clear() - this = " << this << std::endl;
    #endif
%(clear_all_container_code)s

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
            cudaMemcpy(pop%(id)s->%(name)s.data(), pop%(id)s->gpu_%(name)s, pop%(id)s->size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record %(name)s on pop%(id)s failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record %(name)s - [min, max]: " << *std::min_element(pop%(id)s->%(name)s.begin(), pop%(id)s->%(name)s.end() ) << ", " << *std::max_element(pop%(id)s->%(name)s.begin(), pop%(id)s->%(name)s.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->%(name)s.push_back(pop%(id)s->%(name)s);
            else{
                std::vector<%(type)s> tmp = std::vector<%(type)s>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop%(id)s->%(name)s[this->ranks[i]]);
                }
                this->%(name)s.push_back(tmp);
            }
        }""",
    'clear': """
    void clear_%(name)s() {
        for(auto it = this->%(name)s.begin(); it != this->%(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->%(name)s.clear();
    }
""",
    'size_in_bytes': ""
    },
    'semiglobal': { # Does not exist for populations
        'struct': "",
        'init': "",
        'recording': "",
        'clear': "",
        'size_in_bytes': ""
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
            this->%(name)s.push_back(pop%(id)s->%(name)s);
        } """,
    'clear': """
    void clear_%(name)s() {
        this->%(name)s.clear();
    }
    """,
    'size_in_bytes': ""
    }
}

cuda_projection = {
    'struct': """
class ProjRecorder%(id)s : public Monitor
{
protected:
    int _id;
    std::vector<int> indices;

public:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder%(id)s (" << this << ") instantiated." << std::endl;
    #endif
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj%(id)s->get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        // initialize container    
%(init_code)s

        // add monitor to global list
        this->_id = addRecorder(static_cast<Monitor*>(this));
    #ifdef _DEBUG
        std::cout << "ProjRecorder%(id)s (" << this << ") received list position (ID) = " << this->_id << std::endl;
    #endif
    };

    void record() {
%(recording_code)s
    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjMonitor%(id)s::clear()." << std::endl;
    #endif
%(clear_container_code)s
    }

%(clear_individual_code)s

%(struct_code)s
};
""",
    'local': {
        'struct': """
    // Local variable %(name)s
    std::vector< std::vector< std::vector< %(type)s > > > %(name)s ;
    bool record_%(name)s ;
""",
    'init': """
        // Local variable %(name)s
        this->%(name)s = std::vector< std::vector< std::vector< %(type)s > > >();
        this->record_%(name)s = false;
""",
    'recording': """
        // Local variable %(name)s
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            auto host_local = proj%(id)s->get_device_matrix_variable_as_lil<%(float_prec)s>(proj%(id)s->gpu_%(name)s);

            std::vector< std::vector< double > > tmp;
            for (auto idx = 0; idx < this->ranks.size(); idx++) {
                tmp.push_back(host_local[this->indices[idx]]);
            }

            this->%(name)s.push_back(tmp);
            tmp.clear();
        }
""",
    'clear': """
    void clear_%(name)s() {
        // local variable %(name)s
        for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        %(name)s.clear();
        %(name)s.shrink_to_fit();
    }
""",
        'size_in_bytes': """
// local variable %(name)s
size_in_bytes += sizeof(std::vector<%(type)s>) * %(name)s.capacity();
for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
    size_in_bytes += sizeof(%(type)s) * it->capacity();
}
"""
    },
    'semiglobal': {
        'struct': """
    // Semiglobal variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ;
""",
    'init': """
        // Semiglobal variable %(name)s
        this->%(name)s = std::vector< std::vector< %(type)s > > ( this->ranks.size(), std::vector<%(type)s>() );
        this->record_%(name)s = false;
""",
    'recording': """
        // Semiglobal variable %(name)s
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ ) ) {
            auto data = std::vector<%(type)s>(proj%(id)s->nb_dendrites(), 0.0);
            cudaMemcpy( data.data(), proj%(id)s->gpu_%(name)s, proj%(id)s->nb_dendrites() * sizeof(%(type)s), cudaMemcpyDeviceToHost);

        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess )
                std::cout << "record %(name)s on proj%(id)s failed: " << cudaGetErrorString(err) << std::endl;
        #endif

            for ( int i = 0; i < this->ranks.size(); i++) {
                this->%(name)s[i].push_back(data[this->ranks[i]]);
            }
        }
""",
        'clear': """
    void clear_%(name)s() {
        // semiglobal variable %(name)s
        for (auto it = %(name)s.begin(); it != %(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        %(name)s.clear();
        %(name)s.shrink_to_fit();
    }
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
    'init': """
        // Global variable %(name)s
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false;
""",
    'recording': """
        // Global variable %(name)s
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            %(type)s tmp = %(type)s(0);
            cudaMemcpy( &tmp, proj%(id)s->gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);

            this->%(name)s.push_back(tmp);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess )
                std::cout << "record %(name)s on proj%(id)s failed: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
        'clear': """
    void clear_%(name)s() {
        // global variable %(name)s
        %(name)s.clear();
        %(name)s.shrink_to_fit();
    }
""",
        'size_in_bytes': """
// global variable %(name)s
size_in_bytes += sizeof(%(type)s) * %(name)s.capacity();
"""
    }
}
