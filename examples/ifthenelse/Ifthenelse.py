from ANNarchy4 import *

#
# Define the neuron classes
#
Input = Neuron(   tau = 1.0,
                  noise = 0.5,
                  baseline = Variable(init = 0.0),
                  mp = Variable(eq = """
                  tau * dmp/dt + mp = baseline + noise * RandomDistribution('uniform', [-1,1]) """),
                  rate = Variable(eq = "rate = pos(mp)"),
                  order = ['mp','rate'] 
               )

Focus = Neuron( tau = 20.0,
                noise = 0.0,
                baseline = 0.0,
                threshold_min = 0.0,
                threshold_max = 1.0,
                mp = Variable(eq = """
                tau * dmp/dt + mp = sum(exc) - sum(inh) + baseline 
                                        + noise * RandomDistribution('uniform', [-1,1])"""),
                rate = Variable(eq = """
                rate = if not mp > 0  
                            then 0
                            else 
                                if mp < 1 
                                    then mp 
                                    else 1 
                        """, init = 0.0),
                order = ['mp', 'rate']
	       )
		
				
InputPop = Population("Input", (20,20,1), Input)
FocusPop = Population("Focus", (20,20,1), Focus)

Proj1 = Projection( pre = InputPop, post = "Focus", target = 'exc', connector = Connector( type='One2One', weights=RandomDistribution('constant', [1.0]) ) )

Proj2 = Projection( pre = "Focus", post = "Focus", target = 'inh', connector = Connector( type='DoG', weights=RandomDistribution('uniform', [0,1]), amp_pos=0.2, sigma_pos=0.1, amp_neg=0.1, sigma_neg=0.7 ) )

#
# Analyse and compile everything, initialize the parameters/variables...
#
Compile(debugBuild=True)


Simulate(10)
