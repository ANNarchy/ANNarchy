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
record_base_class = """
/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, long int offset){
        this->ranks = ranks;
        this->period = period;
        this->offset = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    virtual void record() = 0;
    virtual void record_targets() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period;
    long int offset;

};
%(record_classes)s
"""

omp_population = {
    'template': """
class PopRecorder%(id)s : public Monitor
{
public:
    PopRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset) {
%(init_code)s
    }

    void record() {
%(recording_code)s
    }

    void record_targets() {
%(recording_target_code)s
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            if(!this->partial)
                this->%(name)s.push_back(pop%(id)s.%(name)s); 
            else{
                std::vector<%(type)s> tmp = std::vector<%(type)s>();
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop%(id)s.%(name)s[this->ranks[i]]);
                }
                this->%(name)s.push_back(tmp);
            }
        }"""
    },
    'semiglobal': { # Does not exist for populations
        'struct': "", 
        'init': "",
        'recording': ""
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(pop%(id)s.%(name)s); 
        } """    
    }
}

cuda_population = {
    'template': """
class PopRecorder%(id)s : public Monitor
{
public:
    PopRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    void record() {
%(recording_code)s
    };
    void record_targets() {
%(recording_target_code)s
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
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
                for(int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop%(id)s.%(name)s[this->ranks[i]]);
                }
                this->%(name)s.push_back(tmp);
            }
        }"""
    },
    'semiglobal': { # Does not exist for populations
        'struct': "", 
        'init': "",
        'recording': ""
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(pop%(id)s.%(name)s); 
        } """    
    }
}

omp_projection = {
    'struct': """
class ProjRecorder%(id)s : public Monitor
{
public:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    void record() {
%(recording_code)s
    };
    void record_targets() { /* nothing to do here */ }
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            std::vector< std::vector< %(type)s > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(proj%(id)s.%(name)s[this->ranks[i]]);
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            std::vector< %(type)s > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(proj%(id)s.%(name)s[this->ranks[i]]);
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(proj%(id)s.%(name)s);
        }
"""
    }
}

cuda_projection = {
    'struct': """
class ProjRecorder%(id)s : public Monitor
{
public:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    void record() {
%(recording_code)s
    };
    void record_targets() { /* nothing to do here */ }
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            auto flat_data = std::vector<%(type)s>(proj%(id)s.overallSynapses, 0.0);
            cudaMemcpy( flat_data.data(), proj%(id)s.gpu_%(name)s, proj%(id)s.overallSynapses * sizeof(%(type)s), cudaMemcpyDeviceToHost);

        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record %(name)s on proj%(id)s failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        #endif

            for ( int i = 0; i < this->ranks.size(); i++) {
                auto deflat_data = proj%(id)s.deFlattenDendrite<%(type)s>(flat_data, this->ranks[i] );
                this->%(name)s[i].push_back(deflat_data);

        #ifdef _DEBUG
            std::cout << "record %(name)s - " << this->ranks[i] << " - [min, max]: "
                      << *std::min_element(deflat_data.begin(), deflat_data.end() ) << ", "
                      << *std::max_element(deflat_data.begin(), deflat_data.end() ) << std::endl;
        #endif
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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 ) ) {
            auto data = std::vector<%(type)s>(proj%(id)s.size, 0.0);
            cudaMemcpy( data.data(), proj%(id)s.gpu_%(name)s, proj%(id)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);

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
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
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
    'openmp' : """
        if(this->record_spike){
            for(int i=0; i<pop%(id)s.spiked.size(); i++){
                if(!this->partial){
                    this->spike[pop%(id)s.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop%(id)s.spiked[i])!=this->ranks.end() ){
                        this->spike[pop%(id)s.spiked[i]].push_back(t);
                    }
                }
            }
        } """,
    'cuda' : """if(this->record_spike){
        for(int i=0; i<pop%(id)s.%(spiked_size)s; i++){
            %(skip_not_spiked)s

            if(!this->partial){
                this->spike[%(spiked_idx)s].push_back(t);
            }
            else{
                if( std::find(this->ranks.begin(), this->ranks.end(), %(spiked_idx)s)!=this->ranks.end() ){
                    this->spike[%(spiked_idx)s].push_back(t);
                }
            }
        }
    } """
}
