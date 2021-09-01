#===============================================================================
#
#     MonitorTemplate.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

    ~Monitor() = default;

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
%(determine_size)s
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "Delete instance of PopRecorder%(id)s ( " << this << " ) " << std::endl;
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
        std::cout << "PopMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {
%(clear_monitor_code)s
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
        for(auto it = this->%(name)s.begin(); it != this->%(name)s.end(); it++)
            it->clear();
        this->%(name)s.clear();
"""
    },
    'semiglobal': { # Does not exist for populations
        'struct': "",
        'init': "",
        'recording': "",
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
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor%(id)s::clear(): not implemented for openMP paradigm." << std::endl;
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
    }
}
