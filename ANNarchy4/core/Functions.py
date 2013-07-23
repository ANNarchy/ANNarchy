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

    header += '\n//Main simulator file\n'
    header += '#include "ANNarchy.h"\n'
    header += '\n#endif'

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
            code += 'net_->connect('+str(p.pre.id)+', '+str(p.post.id)+', '+p.connector.genCPPCall() +', '+ str(p.projClass['ID'])+');\n'

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

def Compile(cppStandAlone=False, debugBuild=False):
    #
    # create build directory
    if os.path.exists(annarchy_dir):
        shutil.rmtree(annarchy_dir, True)
    
    os.mkdir(annarchy_dir)
    os.mkdir(annarchy_dir+'/pyx')
    os.mkdir(annarchy_dir+'/build')

    sources_dir = os.path.abspath(os.path.dirname(__file__)+'/../data')

    for f in os.listdir(sources_dir):
        if not os.path.isdir(os.path.abspath(sources_dir+'/'+f)):
            shutil.copy(sources_dir+'/'+f, annarchy_dir)
    for f in os.listdir(sources_dir+'/cpp'):
        shutil.copy(sources_dir+'/cpp/'+f, annarchy_dir+'/build/'+f)
    for f in os.listdir(sources_dir+'/pyx'):
        shutil.copy(sources_dir+'/pyx/'+f, annarchy_dir+'/pyx/'+f)

    sys.path.append(annarchy_dir)

    #
    # code generation
    print '\nGenerate files\n'
    for pop in populations_:
        pop.generate()

    for proj in projections_:
        proj.generate()

    createIncludes()

    updateANNarchyClass(cppStandAlone)

    genANNarchyPyx()

    #
    # create ANNarchyCore.so and py extensions
    print '\nStart compilation ...\n'
    if sys.platform.startswith('linux'):
        import subprocess
        os.chdir(annarchy_dir)

        #
        # apply +x lost by copy
        os.system('chmod +x compile*')

        if not debugBuild:
            p = subprocess.Popen(['./compile.sh'])
        else:
            p = subprocess.Popen(['./compiled.sh'])

        p.wait()
        os.chdir('..')

        #
        # bind the py extensions to the corresponding python objects
        import ANNarchyCython
        for p in populations_:
            p.cyInstance = eval('ANNarchyCython.py'+p.name.capitalize()+'()')

        #
        # instantiate projections
        for p in projections_:
            conn = p.connector.instantiateConnector(p.connector.type)          
            p.cyInstance = conn.connect(p.pre, p.post, p.connector.weights, p.post.targets.index(p.target), p.connector.parameters)

    else:
        print 'automated compilation and cython/python binding only available under linux currently.'
        import subprocess
        p = subprocess.Popen(['compile.bat'])
        p.wait()

    print '\nCompilation process done.\n'
    
def Simulate(duration):
    import ANNarchyCython
    ANNarchyCython.pyNetwork().Run(duration)
