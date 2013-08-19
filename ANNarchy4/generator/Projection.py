"""
Projection generator.
"""

from ANNarchy4 import parser
from ANNarchy4.core import Global

class Projection:
    """
    Projection generator class.
    """
    def __init__(self, synapse):
        """
        Projection generator class constructor.
        """
        self.synapse = synapse
        self.parsed_synapse_variables = []
        
        #
        # for each synpase we create an own projection type
        if self.synapse:
            id   = len(Global.generatedProj_)+1
            name = 'Projection'+str(id)

            self.hFile = Global.annarchy_dir+'/build/'+name+'.h'
            self.cppFile = Global.annarchy_dir+'/build/'+name+'.cpp'

            Global.generatedProj_.append( { 'name': name, 'ID': id } )
            self.projClass = { 'class': 'Projection', 'ID': id }
            
            synapseParser = parser.SynapseAnalyser(self.synapse.variables)
            self.parsed_synapse_variables = synapseParser.parse()

        else:
            self.projClass = { 'class': 'Projection', 'ID': 0 }

    def generate(self):
        """
        generate projection c++ code.
        """
        def member_def(parsed_variables):
            """
            create variable/parameter header entries.
            """
            code = ''
            for v in parsed_variables:
                if v['name']== 'psp':
                    continue
                    
                pre_def = ['dt','tau','value']
                if v['name'] in pre_def:
                    continue
                    
                if v['type']=='parameter':
                    code += "\tDATA_TYPE "+v['name']+"_;\n"
                elif v['type']=='local':
                    code += "\tstd::vector<DATA_TYPE> "+v['name']+"_;\n"
                else: # global (postsynaptic neurons), or weight bound
                    code += "\tDATA_TYPE "+v['name']+"_;\n"

            return code

        def init(parsed_variables):
            """
            create variable/parameter constructor entries.
            """
            code = ''
            for v in parsed_variables:
                if v['name'] == 'value':
                    continue
                        
                code += "\t"+v['init']+"\n"

            return code
            
        def compute_sum(parsed_variables):
            """
            update the weighted sum code.
            
            TODO: delay mechanism
            """

            #
            # check if 'psp' is contained in variable set
            psp_code= ''
            
            for v in parsed_variables:
                if v['name'] == 'psp':
                    psp_code = v['cpp'].split('=')[1]
                    
            if len(psp_code) > 0:
                code = '''\tDATA_TYPE psp = 0.0;
\tfor(int i=0; i<(int)value_.size();i++) {
\t\tpsp += %(pspCode)s
\t}
\tsum_ = psp;''' % { 'pspCode': psp_code } 
            else:
                code = 'Projection::computeSum();'

            return code
            
        def local_learn(parsed_variables):
            """
            generate synapse update per pre neuron
            """

            code= ''
            loop= ''

            if self.synapse.order==[]:
                for v in parsed_variables:
                    if v['name']=='psp':
                        continue
                    if v['type'] == 'global':
                        continue

                    if len(v['cpp'])>0:
                        loop +='\t\t'+v['cpp']+'\n'
                       
            else:
                for v in self.synapse.order:
                    if v=='psp':
                        continue
                    if v == 'global':
                        continue

                    for v2 in parsed_variables:
                        if v == v2['name']:
                            if len(v2['cpp'])>0:
                                loop +='\t\t'+v2['cpp']+'\n'
            

            code = '\tfor(int i=0; i<(int)rank_.size();i++) {\n'
            code += loop
            code += '\t}\n'

            return code

        def global_learn(parsed_variables):
            """
            generate synapse update per post neuron
            """
            
            code= ''
            loop = ''
            for v in parsed_variables:
                if v['name']=='psp':
                    continue
                if v['type'] == 'local':
                    continue
                
                if len(v['cpp'])>0:
                    loop +='\t\t'+v['cpp']+'\n'

            code += loop
            return code

        #
        # generate func body            
        if self.synapse:
            name = self.projClass['class']+str(self.projClass['ID'])

            header = '''#ifndef __%(name)s_H__
#define __%(name)s_H__

#include "Global.h"

class %(name)s : public Projection {
public:
%(name)s(Population* pre, Population* post, int postRank, int target);

%(name)s(int preID, int postID, int postRank, int target);

~%(name)s();

void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());

void computeSum();

void globalLearn();

void localLearn();

private:
%(synapseMember)s
};
#endif
''' % { 'name': name, 
        'synapseMember': member_def(self.parsed_synapse_variables) }

            body = '''#include "%(name)s.h"
%(name)s::%(name)s(Population* pre, Population* post, int postRank, int target) : Projection(pre, post, postRank, target) {
%(init)s
}

%(name)s::%(name)s(int preID, int postID, int postRank, int target) : Projection(preID, postID, postRank, target) {
%(init)s
}

%(name)s::~%(name)s() {

}

void %(name)s::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
    Projection::initValues(rank, value, delay);
}

void %(name)s::computeSum() {
%(sum)s
}

void %(name)s::localLearn() {
%(local)s
}

void %(name)s::globalLearn() {
%(global)s
}

''' % { 'name': name, 
        'init': init(self.parsed_synapse_variables), 
        'sum': compute_sum(self.parsed_synapse_variables), 
        'local': local_learn(self.parsed_synapse_variables), 
        'global': global_learn(self.parsed_synapse_variables) }

            with open(self.hFile, mode = 'w') as w_file:
                w_file.write(header)

            with open(self.cppFile, mode = 'w') as w_file:
                w_file.write(body)

