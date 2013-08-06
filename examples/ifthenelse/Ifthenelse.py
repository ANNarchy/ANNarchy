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
	       
Oja = Synapse(tau = 5000,
              alpha = Variable(init=0.1, eq = "alpha = post.rate"),
              value = Variable(init=0.0, eq = "dvalue / dt = alpha * (pre.rate*post.rate - post.rate*post.rate*value)"),
              psp = Variable(eq = "psp=(1-pre.rate)*value")

              )
		
				
InputPop = Population("Input", (20,20,1), Input)
FocusPop = Population("Focus", (20,20,1), Focus)

Proj1 = Projection( pre = InputPop, post = "Focus", target = 'exc', synapse=Oja, connector = Connector( conn_type='One2One', weights=RandomDistribution('constant', [1.0]) ) )

Proj2 = Projection( pre = "Focus", post = "Focus", target = 'inh', connector = Connector( conn_type='DoG', weights=RandomDistribution('uniform', [0,1]), amp_pos=0.2, sigma_pos=0.1, amp_neg=0.1, sigma_neg=0.7 ) )

#
# Analyse and compile everything, initialize the parameters/variables...
#
compile(debug_build=True)

simulate(10)
