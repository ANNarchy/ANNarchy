"""
    RecordGenerator.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import BaseTemplate

class RecordGenerator:
    def __init__(self, annarchy_dir, populations, projections, net_id):
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._net_id = net_id
        
    def generate(self):
        record_class = ""
        for pop in self._populations:
            record_class += self._pop_recorder_class(pop)

        for proj in self._projections:
            record_class += self._proj_recorder_class(proj)
        
        code = BaseTemplate.monitor % {'record_classes': record_class} 
        
        # Generate header code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/Recorder.h', 'w') as ofile:
            ofile.write(code)

    def _pop_recorder_class(self, pop):
        tpl_code = """
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
""" 
        init_code = ""
        recording_code = ""
        struct_code = ""

        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                struct_code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ; """ % {'type' : var['ctype'], 'name': var['name']}
                init_code += """
        this->%(name)s = std::vector< std::vector< %(type)s > >();
        this->record_%(name)s = false; """ % {'type' : var['ctype'], 'name': var['name']}
                recording_code += """
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
        }""" % {'id': pop.id, 'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                struct_code += """
    // Global variable %(name)s
    std::vector< %(type)s > %(name)s ;
    bool record_%(name)s ; """ % {'type' : var['ctype'], 'name': var['name']}
                init_code += """
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false; """ % {'type' : var['ctype'], 'name': var['name']}
                recording_code += """
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(pop%(id)s.%(name)s); 
        } """ % {'id': pop.id, 'type' : var['ctype'], 'name': var['name']}
        
        if pop.neuron_type.type == 'spike':
            struct_code += """
    // Local variable %(name)s
    std::map<int, std::vector< long int > > %(name)s ;
    bool record_%(name)s ; """ % {'type' : 'long int', 'name': 'spike'}
            init_code += """
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop%(id)s.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; """ % {'id': pop.id, 'type' : 'long int', 'name': 'spike'}
            recording_code += """
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
        }""" % {'id': pop.id, 'type' : 'int', 'name': 'spike'}

        return tpl_code % {'id': pop.id, 'init_code': init_code, 'recording_code': recording_code, 'struct_code': struct_code}

    def _proj_recorder_class(self, proj):
        """
        Generate the code for the recorder object.

        Templates:

            record
        """
        tpl_code = ProjTemplate.record
        init_code = ""
        recording_code = ""
        struct_code = ""

        for var in proj.synapse.description['variables']:
            struct_code += tpl_code[var['locality']]['struct'] % {'type' : var['ctype'], 'name': var['name']}
            init_code += tpl_code[var['locality']]['init'] % {'type' : var['ctype'], 'name': var['name']}
            recording_code += tpl_code[var['locality']]['recording'] % {'id': proj.id, 'type' : var['ctype'], 'name': var['name']}

        return tpl_code['struct'] % {'id': proj.id, 'init_code': init_code, 'recording_code': recording_code, 'struct_code': struct_code}
