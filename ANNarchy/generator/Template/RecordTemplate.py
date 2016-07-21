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

    virtual void record() {std::cout << "recording" << std::endl;};

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
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    virtual void record() {
%(recording_code)s
    };
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
    virtual void record() {
%(recording_code)s
    };
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
            std::cout << "record %(name)s."<< std::endl;
            std::cout << "  [min, max]: " << *std::min_element(pop%(id)s.%(name)s.begin(), pop%(id)s.%(name)s.end() ) << ", " << *std::max_element(pop%(id)s.%(name)s.begin(), pop%(id)s.%(name)s.end() ) << std::endl;
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
    'cuda' : """
        if(this->record_spike){
            unsigned int num_events = 0;
            cudaMemcpy(&num_events, pop%(id)s.gpu_num_events, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            if (num_events > 0) {
                pop%(id)s.spiked = std::vector<int>(num_events, 0);
                cudaMemcpy(pop%(id)s.spiked.data(), pop%(id)s.gpu_spiked, num_events*sizeof(int), cudaMemcpyDeviceToHost);

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
            }
        } """
}