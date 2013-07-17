import Global
import os, sys
from Master import Master

class Synapse(Master):
    def __init__(self, debug=False, order=[], **keyValueArgs):
        if debug==True:
            print '\n\tSynapse class\n'
        
        Master.__init__(self, debug, order, keyValueArgs)

class Projection:
    """
    Definition of a projection.
    """
    def __init__(self, pre, post, target, connector=None, synapse=None, learningRule=None, debug=False):
        """
        * pre: pre synaptic layer (either string or object)
        * post: post synaptic layer (either string or object)
        * target: connection type
        * synapse: synapse object
        * learningRule: learning rule object
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
        self.learningRule = learningRule

        self.projClass = None
        self.cyInstances=[]

        if self.synapse or self.learningRule:
            id   = len(Global.generatedProj_)+1
            name = 'Projection'+str(id)

            self.hFile = os.getcwd()+'/build/'+name+'.h'
            self.cppFile = os.getcwd()+'/build/'+name+'.cpp'

            Global.generatedProj_.append( { 'name': name, 'ID': id } )
            self.generate(name)

            self.projClass = { 'class': name, 'ID': id }
        else:
            self.projClass = { 'class': 'Projection', 'ID': 0 }

        Global.projections_.append(self)

    def generateMemberDef(self):
        code = ''
        for v in self.synapse.variables:
            code += "\tstd::vector<DATA_TYPE>"+v['name']+"_;\n"

        return code

    def generateInit(self):
        code = ''
        for v in self.synapse.variables:
            if v['init']:
                code += "\t"+v['name']+"_ = std::vector<DATA_TYPE>(rank.size(), "+str(v['init'])+");\n"

        return code

    def generateComputeSum(self):
        code = 'DATA_TYPE psp = 0.0;\n'
        code+= '\tfor(int i=0; i<(int)value_.size();i++) {\n'

        for v in self.synapse.variables:
            if v['name'] == 'psp':
                rSide = v['eq'].split('=')[1]

                rSide = rSide.replace('value', 'value_[i]')
                rSide = rSide.replace('pre.rate', '(*pre_rates_)[i]')

                for v2 in self.synapse.variables:
                    rSide = rSide.replace(v2['name'], v2['name']+'_[i]')

                code += '\t\tpsp += '+rSide+';\n';
                code += '\t}'

        return code;

    def generate(self):
        if self.synapse or self.learningRule:
            name = self.projClass['class']+str(self.projClass['ID'])

            header = '''#ifndef __%(name)s_H__
#define __%(name)s_H__

#include "Global.h"

class %(name)s : public Projection {
public:
%(name)s(Population* pre, Population* post, int postRank) : Projection(pre, post, postRank) {

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
