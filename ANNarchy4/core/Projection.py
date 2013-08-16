import Global
import os, sys
import Master

from ANNarchy4 import parser

class Projection:
    """
    Definition of a projection.
    """
    def __init__(self, pre, post, target, connector=None, synapse=None, debug=False):
        """
        * pre: pre synaptic layer (either string or object)
        * post: post synaptic layer (either string or object)
        * target: connection type
        * synapse: synapse object
        """
        
        if debug == True:
            print 'Create projection ...'
        
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object 
        if isinstance(pre, str):
            for p in Global.populations_:
                if p.name == pre:
                    self.pre = p
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for p in Global.populations_:
                if p.name == post:
                    self.post = p
        else:
            self.post= post
            
        self.post.generator.add_target(target)
        self.target = target
        self.connector = connector
        self.synapse = synapse

        self.projClass = None

        if self.synapse:
            id   = len(Global.generatedProj_)+1
            name = 'Projection'+str(id)

            self.hFile = Global.annarchy_dir+'/build/'+name+'.h'
            self.cppFile = Global.annarchy_dir+'/build/'+name+'.cpp'

            Global.generatedProj_.append( { 'name': name, 'ID': id } )
            self.projClass = { 'class': 'Projection', 'ID': id }
            
            synapseParser = parser.SynapseAnalyser(self.synapse.variables)
            self.parsedSynapseVariables = synapseParser.parse()

        else:
            self.projClass = { 'class': 'Projection', 'ID': 0 }

        Global.projections_.append(self)

    def generate_cpp_add(self):
        if self.connector != None:
            return ('net_->connect('+
                str(self.pre.id)+', '+
                str(self.post.id)+', '+
                self.connector.cpp_call() +', '+ 
                str(self.projClass['ID'])+', '+ 
                str(self.post.generator.targets.index(self.target)) +
                ');\n')
        else:
            print '\tWARNING: no connector object provided.'
            return ''
        
    def generateMemberDef(self):
        code = ''
        for v in self.parsedSynapseVariables:
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

    def generateInit(self):
        code = ''
        if self.synapse:
            for v in self.parsedSynapseVariables:
                if v['name'] == 'value':
                    continue
                    
                code += "\t"+v['init']+"\n"

        return code

    def generateComputeSum(self):

        #
        # check if 'psp' is contained in variable set
        pspCode= ''
        
        for v in self.parsedSynapseVariables:
            if v['name'] == 'psp':
                pspCode = v['cpp'].split('=')[1]
                
        if len(pspCode) > 0:
            code = '\tDATA_TYPE psp = 0.0;\n'
            code+= '\tfor(int i=0; i<(int)value_.size();i++) {\n'
            code+= '\t\tpsp +='+pspCode+'\n' 
            code+= '\t}\n'
            code+= '\tsum_ = psp;\n'
        else:
            code = 'Projection::computeSum();'

        return code
    
    def generateLocalLearn(self):

        code= ''
        loop= ''

        if self.synapse.order==[]:
            for v in self.parsedSynapseVariables:
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

                for v2 in self.parsedSynapseVariables:
                    if v == v2['name']:
                        if len(v2['cpp'])>0:
                            loop +='\t\t'+v2['cpp']+'\n'
        

        code = '\tfor(int i=0; i<(int)rank_.size();i++) {\n'
        code += loop
        code += '\t}\n'

        return code

    def generateGlobalLearn(self):

        code= ''
        loop = ''
        for v in self.parsedSynapseVariables:
            if v['name']=='psp':
                continue
            if v['type'] == 'local':
                continue
            
            if len(v['cpp'])>0:
                loop +='\t\t'+v['cpp']+'\n'

        code += loop
        return code

    def generate(self):
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
''' % { 'name': name, 'synapseMember':self.generateMemberDef() }

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

''' % { 'name': name, 'init':self.generateInit(), 'sum':self.generateComputeSum(), 'local': self.generateLocalLearn(), 'global': self.generateGlobalLearn() }

            with open(self.hFile, mode = 'w') as w_file:
                w_file.write(header)

            with open(self.cppFile, mode = 'w') as w_file:
                w_file.write(body)
        
    def compile(self):
        print '\tCompile '+ self.__class__.__name__        
