from Global import *
import os, sys
from ANNarchy4 import parser
import re
from Random import *
from Variable import *
import Functions

class Population(object):
    def __init__(self, name, geometry, neuron, debug=False):
        self.geometry = geometry
        self.neuron = neuron
        self.name = name
        self.id = len(populations_)
        self.targets = []

        self.header = Functions.annarchy_dir+'/build/'+self.name+'.h'
        self.body = Functions.annarchy_dir+'/build/'+self.name+'.cpp'
        self.pyx = Functions.annarchy_dir+'/pyx/'+self.name+'.pyx'

        populations_.append(self)

    def getSize(self):
        size = 1

        for i in xrange(len(self.geometry)):
            size *= self.geometry[i];

        return size

    def getWidth(self):
        return self.geometry[0]

    def getHeight(self):
        return self.geometry[1]    

    def getDepth(self):
        return self.geometry[2]

    def getRankFromCoordinate(self, w, h, d):
        return d*self.geometry[0]*self.geometry[1] + self.geometry[0] * h + w

    def getCoordinateFromRank(self, rank):
        coord = [0,0,0]

        if(self.geometry[2]==1):
            if(self.geometry[1]==1):
                coord[0] = rank
            else:
                coord[1] = rank / self.geometry[0]
                coord[0] = rank - coord[1]*self.geometry[1]
        else:
             coord[2] = rank / ( self.geometry[0] * self.geometry[1] )

             #coord in plane
             pRank = rank - coord[2] * ( self.geometry[0] * self.geometry[1] )
             coord[1] = pRank / self.geometry[0]
             coord[0] = pRank - coord[1]*self.geometry[1]

	return coord

    def getNormalizedCoordinateFromRank(self, rank, norm=1):
        coord = [0,0,0]

        if(self.geometry[2]==1):
            if(self.geometry[1]==1):
                coord[0] = rank/(self.getSize()-norm)
            else:
                w = rank / self.geometry[0]
                h = rank - coord[1]*self.geometry[1]
                coord[0] = w / (self.getWidth()-norm)
                coord[1] = h / (self.getHeight()-norm)
        else:
             d = rank / ( self.geometry[0] * self.geometry[1] )
             #coord in plane
             pRank = rank - coord[2] * ( self.geometry[0] * self.geometry[1] )
             h = pRank / self.geometry[0]
             w = pRank - coord[1]*self.geometry[1]

             coord[0] = w / (self.getWidth()-norm)
             coord[1] = h / (self.getHeight()-norm)
             coord[2] = d / (self.getDepth()-norm)

        return coord

    def getName(self):
        return self.name

    def addTarget(self, target):
        self.targets.append(target)

        self.targets = list(set(self.targets))   # little trick to remove doubled entries

    def set(self, **keyValueArgs):
        # update neuron instance definition
        self.neuron.set(keyValueArgs)
    
    def getID(self):
        return self.id

    def generate(self):
        print '\tGenerate files for '+ self.__class__.__name__

        neurVars = self.neuron.variables

        #
        #   remove unset items
        for v in neurVars:
            if 'var' in v:
                if v['var']==None:
                    del v['var']
            if 'init' in v:
                if v['init']==None:
                    del v['init']

        #
        #   replace all RandomDistribution by rand variables with continous numbers
        #   and stores the corresponding call as local variable
        i = 0
        for v in neurVars:
            if 'var' not in v.keys():
                continue 

            if isinstance(v['var'], Variable):
                eq = v['var'].eq

                if eq == None:
                    continue

                if eq.find('RandomDistribution') != -1:
                    tmp = eq.split('RandomDistribution')
                    
                    for t in tmp:
                        if t.find('sum(') != -1:
                            continue

                        phrase = re.findall('\(.*?\)', t) # find the shortest term within to brackets
                        if phrase != []:
                            for p in phrase:
                                neurVars.append({'name':'_rand_'+str(i), 'init': p})
                                eq = eq.replace('RandomDistribution'+p, '_rand_'+str(i))
                            i += 1

                    v['var'].eq = eq
                          
        #
        #   parse neuron
        self.neuronParser = parser.NeuronAnalyser(neurVars, self.targets);
        self.parsedNeuron = self.neuronParser.parse()

        #
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self.generateHeader())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self.generateBody())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self.generatePyx())

    def genConstructor(self):
        constructor = ''

        for v in self.parsedNeuron:
            if '_rand_' in v['name']:   # skip local member
                continue
                
            if 'rate' == v['name']:
                continue

            constructor += '\t'+v['init']+'\n'

        return constructor
    
    def genMember(self):
        member = ''
        
        for v in self.parsedNeuron:
            if '_rand_' in v['name']:   # skip local member
                member += '\tstd::vector<DATA_TYPE> '+v['name']+';\n'
                continue
            if 'rate' == v['name']:
                continue

            member += '\t'+v['def']+'\n'
        
        return member

    def genMetaStep(self):
	meta = ''
	loop = ''

        for v in self.parsedNeuron:
            if '_rand_' in v['name']:   # skip local member
                #for v2 in self.parsedNeuron:
                #    if v['name'] in v2['cpp']:
                #        call = 'RandomDistribution'+v['init'].split('=')[1]
                #        call = call.split(';')[0]
                #        cmd = call+'.genCPP()'
                #        v2['cpp'] = v2['cpp'].replace(v['name'], eval(cmd)+'.getValue()')
                call = ('RandomDistribution'+v['init'].split('=')[1]).replace(';','')
                meta += '\t'+v['name']+'= '+eval(call+'.genCPP()')+'.getValues(nbNeurons_);\n'

        loop += '\tfor(int i=0; i<nbNeurons_; i++) {\n'

        if self.neuron.order == []:
            # order does not play an important role        
 	    for v in self.parsedNeuron:
                if '_rand_' in v['name']:   # skip local member
                    continue

                loop += '\t\t'+v['cpp']+'\n'

            loop+= '\t}'
        else:
            for v in self.neuron.order:
                for v2 in self.parsedNeuron:
                    if '_rand_' in v2['name']:   # skip local member
                        continue

                    if v2['name'] == v:
                        loop += '\t\t'+v2['cpp']+'\n'

            loop+= '\t}'

	import re
        # replace all _rand_.. by _rand_..[i]
	loop = re.sub('_rand_[0-9]+', lambda m: m.group(0)+'[i]', loop)

        # replace all sum(typeID) by 
	loop = loop.replace('sum(', 'sum(i,')

	code = meta + loop
        return code

    def genAccess(self):
        access = ''

        for v in self.parsedNeuron:
            if '_rand_' in v['name']:
                continue

            if v['type'] == 'variable':
                #getter
                access += '\tvoid set'+v['name'].capitalize()+'(std::vector<DATA_TYPE> '+v['name']+') { this->'+v['name']+'='+v['name']+'; }\n'
                access += '\tstd::vector<DATA_TYPE> get'+v['name'].capitalize()+'() { return this->'+v['name']+'; }\n'
            else:
                #setter
                access += '\tvoid set'+v['name'].capitalize()+'(DATA_TYPE '+v['name']+') { this->'+v['name']+'='+v['name']+'; }\n'
                access += '\tDATA_TYPE get'+v['name'].capitalize()+'() { return this->'+v['name']+'; }\n'

        return access

    def genPyxFunc(self):
        code = ''

        for v in self.parsedNeuron:
            if '_rand_' in v['name']:
                continue

            if v['type'] == 'variable':
                #get
                code += '\t\tvector[float] get'+v['name'].capitalize()+'()\n\n'
                #set
                code += '\t\tvoid set'+v['name'].capitalize()+'(vector[float] values)\n\n'
            else:
                code += '\t\tfloat get'+v['name'].capitalize()+'()\n\n'
                code += '\t\tvoid set'+v['name'].capitalize()+'(float value)\n\n'

        return code

    def genPyFunc(self):
        code = ''

        for v in self.parsedNeuron:
            if '_rand_' in v['name']:
                continue

	    code+= '\tproperty '+v['name']+':\n'
            if v['type'] == 'variable':
                #getter
                code += '\t\tdef __get__(self):\n'
                code += '\t\t\treturn np.array(self.cInstance.get'+v['name'].capitalize()+'())\n\n'

            else:
		#getter
                code += '\t\tdef __get__(self):\n'
                code += '\t\t\treturn self.cInstance.get'+v['name'].capitalize()+'()\n\n'
            
	    code+= '\t\tdef __set__(self, value):\n'
            code+= '\t\t\tself.cInstance.set'+v['name'].capitalize()+'(value)\n'

        return code

    def generateHeader(self):
        header = """#ifndef __ANNarchy_%(class)s_H__
#define __ANNarchy_%(class)s_H__

#include "Global.h"

class %(class)s: public Population
{
public:
    %(class)s(std::string name, int nbNeurons);
    
    void metaSum();
    
    void metaStep();
    
    void metaLearn();
    
%(access)s
private:
%(member)s
};
#endif
""" % { 'class': self.name, 'member':self.genMember(), 'access' : self.genAccess() } 

        return header
    
    def generateBody(self):
        body = """#include "%(class)s.h"

%(class)s::%(class)s(std::string name, int nbNeurons):Population(name, nbNeurons)
{
#ifdef _DEBUG
    std::cout << "%(class)s::%(class)s called." << std::endl;
#endif
%(construct)s
    Network::instance()->addPopulation(this);
}

void %(class)s::metaSum() {

    #pragma omp parallel for schedule(dynamic, 10)
    for(int n=0; n<nbNeurons_; n++) {
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->computeSum();
        }
    }
    
}

void %(class)s::metaStep() {
%(metaStep)s    
}

void %(class)s::metaLearn() {

    #pragma omp parallel for schedule(dynamic, 10)
    for(int n=0; n<nbNeurons_; n++) {
        for(int p=0; p< (int)projections_[n].size();p++) {
            projections_[n][p]->learn();
        }
    }
    
}
""" % { 'class': self.name, 'construct': self.genConstructor(), 'metaStep': self.genMetaStep() } 

        return body    

    def generatePyx(self):
        pyx='''from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "../build/Input.h":
	cdef cppclass %(name)s:
		%(name)s(string name, int N)

%(cFunction)s
		string getName()

cdef class py%(cname)s:

	cdef %(name)s* cInstance

	def __cinit__(self):
		print 'Instantiate %(name)s'
		self.cInstance = new %(name)s('%(name)s', %(neuron_count)s)

%(pyFunction)s
	def name(self):
		return self.cInstance.getName()

''' % { 'name':self.name, 'cname':self.name, 'neuron_count': self.getSize(), 'cFunction': self.genPyxFunc(), 'pyFunction':self.genPyFunc() }
        return pyx
