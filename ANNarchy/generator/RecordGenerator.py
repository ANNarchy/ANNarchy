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
from ANNarchy.core import Global
import ANNarchy.generator.Template.ProjectionTemplate as ProjTemplate
import ANNarchy.generator.Template.RecordTemplate as RecTemplate

class RecordGenerator:
    """
    Creates the required codes for recording population
    and projection data
    """
    def __init__(self, annarchy_dir, populations, projections, net_id):
        """
        Constructor, stores all required data for later
        following code generation step

        Parameters:

            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *populations*: list of populations
            * *populations*: list of projections
            * *net_id*: unique id for the current network

        """
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._net_id = net_id

    def generate(self):
        """
        Generate one file "Recorder.h" comprising of Monitor base class and inherited
        classes for each Population/Projection.

        Templates:

            record_base_class
        """
        record_class = ""
        for pop in self._populations:
            record_class += self._pop_recorder_class(pop)

        for proj in self._projections:
            record_class += self._proj_recorder_class(proj)

        code = RecTemplate.record_base_class % {'record_classes': record_class}

        # Generate header code for the analysed pops and projs
        with open(self._annarchy_dir+'/generate/net'+str(self._net_id)+'/Recorder.h', 'w') as ofile:
            ofile.write(code)

    def _pop_recorder_class(self, pop):
        """
        Creates population recording class code.

        Returns:

            * complete code as string

        Templates:

            omp_population, cuda_population
        """
        if Global.config['paradigm']=="openmp":
            template = RecTemplate.omp_population
        else:
            template = RecTemplate.cuda_population

        tpl_code = template['template']

        init_code = ""
        recording_code = ""
        struct_code = ""

        # Rate-coded networks also can record the weighted sums
        wsums_list = []
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                wsums_list.append({'ctype': 'double', 'name': '_sum_'+target, 'locality': 'local'})

        for var in pop.neuron_type.description['variables'] + wsums_list:
            struct_code += template[var['locality']]['struct'] % {'type' : var['ctype'], 'name': var['name']}
            init_code += template[var['locality']]['init'] % {'type' : var['ctype'], 'name': var['name']}
            recording_code += template[var['locality']]['recording'] % {'id': pop.id, 'type' : var['ctype'], 'name': var['name']}

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

        Returns:

            * complete code as string

        Templates:

            record
        """
        tpl_code = ProjTemplate.record
        init_code = ""
        recording_code = ""
        struct_code = ""

        for var in proj.synapse_type.description['variables']:
            struct_code += tpl_code[var['locality']]['struct'] % {'type' : var['ctype'], 'name': var['name']}
            init_code += tpl_code[var['locality']]['init'] % {'type' : var['ctype'], 'name': var['name']}
            recording_code += tpl_code[var['locality']]['recording'] % {'id': proj.id, 'type' : var['ctype'], 'name': var['name']}

        return tpl_code['struct'] % {'id': proj.id, 'init_code': init_code, 'recording_code': recording_code, 'struct_code': struct_code}
