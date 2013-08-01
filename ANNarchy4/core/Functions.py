from Global import *
from Population import *
import sys, os
import shutil

annarchy_dir = os.getcwd() + '/annarchy'
 
def createIncludes():
    headerFile = annarchy_dir + '/build/Includes.h'

    header  = '#ifndef __ANNARCHY_INCLUDES_H__\n'
    header += '#define __ANNARCHY_INCLUDES_H__\n\n'

    header += '\n//Population files\n'
    for pop in populations_:
        header += '#include "'+pop.name+'.h"\n'

    header += '\n//Projection files\n'
    for p in generatedProj_:
        header += '#include "'+ p['name'] +'.h"\n'

    header += '#endif'
    with open(headerFile, mode = 'w') as w_file:
        w_file.write(header)

def genPopAdd():
    code = ''
    for p in populations_:
        size = 1
        for i in range(len(p.geometry)):
            size *= p.geometry[i]

        code += '\t\t'+p.name.capitalize()+'* '+p.name.lower()+' = new '+p.name.capitalize()+'("'+p.name.capitalize()+'", '+str(size)+');\n'
 
    return code

def genProjAdd():
    code = ''

    for p in projections_:
        if(p.connector == None):
            print '\tWARNING: no connector object provided.'
        else:
            code += 'net_->connect('+str(p.pre.id)+', '+str(p.post.id)+', '+p.connector.genCPPCall() +', '+ str(p.projClass['ID'])+', '+ str(p.post.targets.index(p.target)) +');\n'

    return code

def genProjClass():
    def genAddCases():
        code = ''
        for p in generatedProj_:
            code += '\t\tcase '+str(p['ID'])+':\n'+ '\t\t\treturn new '+p['name']+'(pre, post, postNeuronRank, target);\n\n'

        return code

    code = '''class createProjInstance {
public:
	createProjInstance() {};

	Projection* getInstanceOf(int ID, Population *pre, Population *post, int postNeuronRank, int target) {
		switch(ID) {
		case 0:
			return new Projection(pre, post, postNeuronRank, target);
%(case)s

		default:
			std::cout << "Unknown typeID" << std::endl;
			return NULL;
		}
	}
};
''' % { 'case': genAddCases() }
    return code

def updateANNarchyClass(cppStandAlone):
    code = ''
    with open(annarchy_dir+'/build/ANNarchy.h', mode = 'r') as r_file:
        for a_line in r_file:
            if a_line.find('//AddProjection') != -1:
                if(cppStandAlone):
                    code += a_line
                    code += genProjAdd()
            elif a_line.find('//AddPopulation') != -1:
                if(cppStandAlone):
                    code += a_line
                    code += genPopAdd()
            elif a_line.find('//createProjInstance') != -1:
                code += a_line
                code += genProjClass()
            else:
                code += a_line

    with open(annarchy_dir+'/build/ANNarchy.h', mode = 'w') as w_file:
        w_file.write(code)

def genANNarchyPyx():
    code = ''

    code +='include \"Network.pyx\"\n'
    code +='include \"Simulation.pyx\"\n'
    code +='include \"Projection.pyx\"\n'
    code +='include \"Connector.pyx\" \n'
    code +='\n'

    for p in populations_:
        code += 'include \"'+p.name+'.pyx\"\n'

    with open(annarchy_dir+'/pyx/ANNarchyCython.pyx', mode = 'w') as w_file:
        w_file.write(code)
    
def Simulate(duration):
    import ANNarchyCython
    ANNarchyCython.pyNetwork().Run(duration)
