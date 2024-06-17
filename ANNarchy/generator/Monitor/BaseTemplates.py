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
