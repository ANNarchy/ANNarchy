import Global
import os, sys
from Master import Master
import Functions

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
                if p.getName() == pre:
                    self.pre = p
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for p in Global.populations_:
                if p.getName() == post:
                    self.post = p
        else:
            self.post= post
            
        self.post.addTarget(target)
        self.target = target
        self.connector = connector
        self.synapse = synapse

        self.projClass = None

        if self.synapse:
            id   = len(Global.generatedProj_)+1
            name = 'Projection'+str(id)

            self.hFile = Functions.annarchy_dir+'/build/'+name+'.h'
            self.cppFile = Functions.annarchy_dir+'/build/'+name+'.cpp'

            Global.generatedProj_.append( { 'name': name, 'ID': id } )
            self.projClass = { 'class': 'Projection', 'ID': id }

        else:
            self.projClass = { 'class': 'Projection', 'ID': 0 }

        Global.projections_.append(self)

    def generateMemberDef(self):
        code = ''
        for v in self.synapse.parsedVariables:
            if v['name']== 'psp':
                continue

            if v['type']=='parameter':
                code += "\tDATA_TYPE "+v['name']+"_;\n"
            elif v['type']=='local':
                code += "\tstd::vector<DATA_TYPE>"+v['name']+"_;\n"
            else: # global (postsynaptic neurons), or weight bound
                code += "\tDATA_TYPE "+v['name']+"_;\n"

        return code

    def generateInit(self):
        code = ''
        if self.synapse:
            for v in self.synapse.parsedVariables:
                if v['init'] and v['type']=='parameter':
                    code += "\t"+v['init']+"\n"
                elif v['init'] and v['type']=='global':
                    code += "\t"+v['init']+"\n"

        return code

    def generateComputeSum(self):

        #
        # check if 'psp' is contained in variable set
        pspCode= ''
        
        for v in self.synapse.variables:
            if v['name'] == 'psp':
                pspCode = v['var'].eq.split('=')[1]
                pspCode = pspCode.replace('pre.rate', '(*pre_rates_)[i]')

                for v2 in self.synapse.variables:
                    pspCode = pspCode.replace(v2['name'], v2['name']+'_[i]')
                
        if len(pspCode) > 0:
            code = '\tDATA_TYPE psp = 0.0;\n'
            code+= '\tfor(int i=0; i<(int)value_.size();i++) {\n'
            code+= '\t\tpsp +='+pspCode+';\n' 
            code+= '\t}\n'
            code+= '\tsum_ = psp;\n'
        else:
            code = 'Projection::computeSums()'

        return code;

    def generate(self):
        if self.synapse:
            name = self.projClass['class']+str(self.projClass['ID'])

            header = '''#ifndef __%(name)s_H__
#define __%(name)s_H__

#include "Global.h"

class %(name)s : public Projection {
public:
%(name)s(Population* pre, Population* post, int postRank, int target) : Projection(pre, post, postRank, target) {

}

~%(name)s() {

}

void initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay = std::vector<int>());

void computeSum();

void learn();

private:
%(synapseMember)s
};
#endif
''' % { 'name': name, 'synapseMember':self.generateMemberDef() }

            body = '''#include "%(name)s.h"

void %(name)s::initValues(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<int> delay) {
    Projection::initValues(rank, value, delay);

%(init)s
}

void %(name)s::computeSum() {
%(sum)s
}

void %(name)s::learn() {

}

''' % { 'name': name, 'init':self.generateInit(), 'sum':self.generateComputeSum() }

            with open(self.hFile, mode = 'w') as w_file:
                w_file.write(header)

            with open(self.cppFile, mode = 'w') as w_file:
                w_file.write(body)
        
    def compile(self):
        print '\tCompile '+ self.__class__.__name__        
