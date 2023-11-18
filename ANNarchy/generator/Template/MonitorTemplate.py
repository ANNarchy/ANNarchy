"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

record_base_class = """#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};
%(record_classes)s
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

cuda_population = {
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
        std::cout << "PopRecorder%(id)s::~PopRecorder%(id)s - this = " << this << std::endl;
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
            cudaMemcpy(pop%(id)s.%(name)s.data(), pop%(id)s.gpu_%(name)s, pop%(id)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record %(name)s on pop%(id)s failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record %(name)s - [min, max]: " << *std::min_element(pop%(id)s.%(name)s.begin(), pop%(id)s.%(name)s.end() ) << ", " << *std::max_element(pop%(id)s.%(name)s.begin(), pop%(id)s.%(name)s.end() ) << std::endl;
            }
        #endif
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
    'clear': """
        for(auto it = this->%(name)s.begin(); it != this->%(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->%(name)s.clear();
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
            this->%(name)s.push_back(pop%(id)s.%(name)s);
        } """,
    'clear': """
        this->%(name)s.clear();
    """,
    'size_in_bytes': ""
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

cuda_projection = {
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
%(init_code)s
    };

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
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "ProjRecorder%(id)s::clear(): not implemented for cuda paradigm." << std::endl;
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
    'init': """
        // Local variable %(name)s
        this->%(name)s = std::vector< std::vector< std::vector< %(type)s > > >(this->ranks.size(), std::vector< std::vector< %(type)s > >());
        this->record_%(name)s = false;
""",
    'recording': """
        // Local variable %(name)s
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            auto host_local = proj%(id)s.get_device_matrix_variable_as_lil<%(float_prec)s>(proj%(id)s.gpu_%(name)s);

            for (auto idx = 0; idx < this->ranks.size(); idx++) {
                this->%(name)s[idx].push_back(host_local[idx]);
            }
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
            auto data = std::vector<%(type)s>(proj%(id)s.nb_dendrites(), 0.0);
            cudaMemcpy( data.data(), proj%(id)s.gpu_%(name)s, proj%(id)s.nb_dendrites() * sizeof(%(type)s), cudaMemcpyDeviceToHost);

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
    'init': """
        // Global variable %(name)s
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false;
""",
    'recording': """
        // Global variable %(name)s
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            %(type)s tmp = %(type)s(0);
            cudaMemcpy( &tmp, proj%(id)s.gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);

            this->%(name)s.push_back(tmp);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess )
                std::cout << "record %(name)s on proj%(id)s failed: " << cudaGetErrorString(err) << std::endl;
        #endif
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

recording_spike_tpl= {
    'struct': """
    // Local variable %(name)s
    std::map<int, std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ;
    void clear_%(name)s() {
        for ( auto it = %(name)s.begin(); it != %(name)s.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
        // (HD: 8th Sep 2023): do not clear the top-level structure, otherwise the return of get_spike()
        //                     will not be as expected: an empty list assigned to the corresponding neuron
        //                     index.
        //%(name)s.clear();
    }
""",
    'init' : """
        this->%(name)s = std::map<int,  std::vector< %(type)s > >();
        if(!this->partial){
            for(int i=0; i<pop%(id)s.size; i++) {
                this->%(name)s[i]=std::vector<%(type)s>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->%(name)s[this->ranks[i]]=std::vector<%(type)s>();
            }
        }
        this->record_%(name)s = false;
""",
    'record' : {
        'openmp' : """
        if(this->record_%(name)s){
            for(int i=0; i<pop%(id)s.%(rec_target)s.size(); i++){
                if(!this->partial){
                    this->%(name)s[pop%(id)s.%(rec_target)s[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop%(id)s.%(rec_target)s[i])!=this->ranks.end() ){
                        this->%(name)s[pop%(id)s.%(rec_target)s[i]].push_back(t);
                    }
                }
            }
        } """,
        'cuda' : """if(this->record_spike){
            for(int i=0; i<pop%(id)s.spike_count; i++){
                if(!this->partial){
                    this->spike[pop%(id)s.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop%(id)s.spiked[i])!=this->ranks.end() ){
                        this->spike[pop%(id)s.spiked[i]].push_back(t);
                    }
                }
            }
        } """
    },
    'size_in_bytes': {
        'openmp' : """
// record spike events
size_in_bytes += sizeof(%(name)s);
for ( auto it = %(name)s.begin(); it != %(name)s.end(); it++ ) {
    size_in_bytes += sizeof(int); // key
    size_in_bytes += sizeof(%(type)s) * (it->second).capacity(); // value
}
        """,
        'cuda': """
        // TODO:
        """
    },
    'clear': {
        'openmp' : """
            for (auto it = this->spike.begin(); it != this->spike.end(); it++) {
                it->second.clear();
                it->second.shrink_to_fit();
            }
            this->spike.clear();
        """,
        'cuda': """
        // TODO:
        """
    }
}
