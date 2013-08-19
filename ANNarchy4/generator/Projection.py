"""
Projection generator.
"""

from ANNarchy4 import parser
from ANNarchy4.core import Global

class Projection:
    """
    Projection generator class.
    """
    def __init__(self, projection, synapse):
        """
        Projection generator class constructor.
        """
        self.projection = projection
        self.synapse = synapse
        self.parsed_synapse_variables = []
        
        #
        # for each synpase we create an own projection type
        if self.synapse:
            sid   = len(Global.generatedProj_)+1
            name = 'Projection'+str(sid)

            self.h_file = Global.annarchy_dir+'/build/'+name+'.h'
            self.cpp_file = Global.annarchy_dir+'/build/'+name+'.cpp'

            Global.generatedProj_.append( { 'name': name, 'ID': sid } )
            self.proj_class = { 'class': 'Projection', 'ID': sid }
            
            synapse_parser = parser.SynapseAnalyser(self.synapse.variables)
            self.parsed_synapse_variables = synapse_parser.parse()

        else:
            self.proj_class = { 'class': 'Projection', 'ID': 0 }

    def generate_cpp_add(self):
        """
        In case of cpp_stand_alone compilation, this function generates
        the connector calls.
        """
        if self.projection.connector != None:
            return ('net_->connect('+
                str(self.projection.pre.id)+', '+
                str(self.projection.post.id)+', '+
                self.projection.connector.cpp_call() +', '+ 
                str(self.proj_class['ID'])+', '+ 
                str(self.projection.post.generator.targets.index(self.projection.target))+
                ');\n')
        else:
            print '\tWARNING: no connector object provided.'
            return ''

    def generate(self):
        """
        generate projection c++ code.
        """
        def member_def(parsed_variables):
            """
            create variable/parameter header entries.
            """
            code = ''
            for var in parsed_variables:
                if var['name'] == 'psp':
                    continue
                    
                pre_def = ['dt', 'tau', 'value']
                if var['name'] in pre_def:
                    continue
                    
                if var['type'] == 'parameter':
                    code += "\tDATA_TYPE "+var['name']+"_;\n"
                elif var['type'] == 'local':
                    code += "\tstd::vector<DATA_TYPE> "+var['name']+"_;\n"
                else: # global (postsynaptic neurons), or weight bound
                    code += "\tDATA_TYPE "+var['name']+"_;\n"

            return code

        def init(parsed_variables):
            """
            create variable/parameter constructor entries.
            """
            code = ''
            for var in parsed_variables:
                if var['name'] == 'value':
                    continue
                        
                code += "\t"+var['init']+"\n"

            return code
            
        def compute_sum(parsed_variables):
            """
            update the weighted sum code.
            
            TODO: delay mechanism
            """

            #
            # check if 'psp' is contained in variable set
            psp_code = ''
            
            for var in parsed_variables:
                if var['name'] == 'psp':
                    psp_code = var['cpp'].split('=')[1]
                    
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

            code = ''
            loop = ''

            if self.synapse.order == []:
                for var in parsed_variables:
                    if var['name'] == 'psp':
                        continue
                    if var['type'] == 'global':
                        continue

                    if len(var['cpp']) > 0:
                        loop += '\t\t'+var['cpp']+'\n'
                       
            else:
                for var in self.synapse.order:
                    if var == 'psp':
                        continue
                    if var == 'global':
                        continue

                    for var2 in parsed_variables:
                        if var == var2['name']:
                            if len(var2['cpp']) > 0:
                                loop += '\t\t'+var2['cpp']+'\n'
            

            code = '\tfor(int i=0; i<(int)rank_.size();i++) {\n'
            code += loop
            code += '\t}\n'

            return code

        def global_learn(parsed_variables):
            """
            generate synapse update per post neuron
            """
            
            code = ''
            loop = ''
            for var in parsed_variables:
                if var['name'] == 'psp':
                    continue
                if var['type'] == 'local':
                    continue
                
                if len(var['cpp']) > 0:
                    loop += '\t\t'+var['cpp']+'\n'

            code += loop
            return code

        #
        # generate func body            
        if self.synapse:
            name = self.proj_class['class']+str(self.proj_class['ID'])

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

            with open(self.h_file, mode = 'w') as w_file:
                w_file.write(header)

            with open(self.cpp_file, mode = 'w') as w_file:
                w_file.write(body)

