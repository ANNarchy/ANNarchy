#===============================================================================
#
#     BoldModels.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Oliver Maith <>, 
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core.Neuron import Neuron

#our current BOLD computation
BoldNeuron = Neuron(
parameters = """
    ea    = 1.0
    tau_s = 1 / 0.665
    tau_f = 1 / 0.412
    E_0   = 0.3424
    tau_0 = 1.0368
    alpha = 0.3215
    V_0   = 0.02
""",
equations = """
    r             = sum(exc)                                                    : init=0
    1000*ds/dt    = ea * r - s / tau_s - (f_in - 1) / tau_f                     : init=0
    1000*df_in/dt = s                                                           : init=1
    E             = 1 - (1 - E_0)**(1 / f_in)                                   : init=0.3424
    1000*dq/dt    = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)              : init=1
    1000*dv/dt    = 1 / tau_0 * (f_in - f_out)                                  : init=1
    f_out         = v**(1 / alpha)                                              : init=1

    k_1           = 7 * E_0
    k_2           = 2
    k_3           = 2 * E_0 - 0.2
    BOLD          = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v)) : init=0
""",
    name = "BoldNeuron",
    description = "BOLD computation from Maith et al. (2020)."
)


#Stephan et al. (2007)
#V_0, tau_0, tau_s, tau_f they don't mention --> use Frsiton et al. 2000
#E_0, epsilon are free --> E_0 use Friston et al. 2000, epsilon use 1.43 (probably used if 1.43 is fixed in Stephan et al. (2007), value from Obata et al. (2004))
BoldNeuron_CBN = Neuron(
parameters = """
    tau_s   = 1.54
    tau_f   = 2.46
    E_0     = 0.34
    tau_0   = 0.98
    alpha   = 0.33
    V_0     = 0.02
    v_0     = 40.3
    TE      = 40/1000.
    epsilon = 1.43
""",
equations = """
    r             = sum(exc)                                                    : init=0
    1000*ds/dt    = r - s / tau_s - (f_in - 1) / tau_f                          : init=0
    1000*df_in/dt = s                                                           : init=1
    E             = 1 - (1 - E_0)**(1 / f_in)                                   : init=0.3424
    1000*dq/dt    = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)              : init=1
    1000*dv/dt    = 1 / tau_0 * (f_in - f_out)                                  : init=1
    f_out         = v**(1 / alpha)                                              : init=1

    k_1           = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2           = 2 * E_0
    k_3           = 1 - epsilon
    BOLD          = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v)) : init=0
""",
    name = "BoldNeuron CBN",
    description = "BOLD computation with classic coefficients and non-linear BOLD equation (Stephan et al., 2007)."
)


BoldNeuron_CBL = Neuron(
parameters = """
    tau_s   = 1.54
    tau_f   = 2.46
    E_0     = 0.34
    tau_0   = 0.98
    alpha   = 0.33
    V_0     = 0.02
    v_0     = 40.3
    TE      = 40/1000.
    epsilon = 1.43
""",
equations = """    
    r             = sum(exc)                                              : init=0
    1000*ds/dt    = r - s / tau_s - (f_in - 1) / tau_f                    : init=0
    1000*df_in/dt = s                                                     : init=1
    E             = 1 - (1 - E_0)**(1 / f_in)                             : init=0.3424
    1000*dq/dt    = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)        : init=1
    1000*dv/dt    = 1 / tau_0 * (f_in - f_out)                            : init=1
    f_out         = v**(1 / alpha)                                        : init=1

    k_1           = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2           = 2 * E_0
    k_3           = 1 - epsilon
    BOLD          = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v)) : init=0
""",
    name = "BoldNeuron CBL",
    description = "BOLD computation with classic coefficients and linear BOLD equation (Stephan et al., 2007)."
)


BoldNeuron_RBN = Neuron(
parameters = """
    tau_s   = 1.54
    tau_f   = 2.46
    E_0     = 0.34
    tau_0   = 0.98
    alpha   = 0.33
    V_0     = 0.02
    v_0     = 40.3
    TE      = 40/1000.
    epsilon = 1.43
    r_0     = 25
""",
equations = """    
    r             = sum(exc)                                                    : init=0
    1000*ds/dt    = r - s / tau_s - (f_in - 1) / tau_f                          : init=0
    1000*df_in/dt = s                                                           : init=1
    E             = 1 - (1 - E_0)**(1 / f_in)                                   : init=0.3424
    1000*dq/dt    = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)              : init=1
    1000*dv/dt    = 1 / tau_0 * (f_in - f_out)                                  : init=1
    f_out         = v**(1 / alpha)                                              : init=1

    k_1           = 4.3 * v_0 * E_0 * TE
    k_2           = epsilon * r_0 * E_0 * TE
    k_3           = 1 - epsilon
    BOLD          = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v)) : init=0
""",
    name = "BoldNeuron RBN",
    description = "BOLD computation with revised coefficients and non-linear BOLD equation (Stephan et al., 2007)."
)


BoldNeuron_RBL = Neuron(
parameters = """
    tau_s   = 1.54
    tau_f   = 2.46
    E_0     = 0.34
    tau_0   = 0.98
    alpha   = 0.33
    V_0     = 0.02
    v_0     = 40.3
    TE      = 40/1000.
    epsilon = 1.43
    r_0     = 25
""",
equations = """    
    r             = sum(exc)                                              : init=0
    1000*ds/dt    = r - s / tau_s - (f_in - 1) / tau_f                    : init=0
    1000*df_in/dt = s                                                     : init=1
    E             = 1 - (1 - E_0)**(1 / f_in)                             : init=0.3424
    1000*dq/dt    = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)        : init=1
    1000*dv/dt    = 1 / tau_0 * (f_in - f_out)                            : init=1
    f_out         = v**(1 / alpha)                                        : init=1

    k_1           = 4.3 * v_0 * E_0 * TE
    k_2           = epsilon * r_0 * E_0 * TE
    k_3           = 1 - epsilon
    BOLD          = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v)) : init=0
""",
    name = "BoldNeuron RBL",
    description = "BOLD computation with revised coefficients and linear BOLD equation (Stephan et al., 2007)."
)


#new standard model
# damped harmonic oscillators, k->timeconstant, c->damping
# CBF --> try k from Friston
# CMRO2 --> faster --> k=k_CBF*2 (therefore scaling of I_CMRO2 --> if same input CMRO2 and CBF same steady-state)
# critical c --> c**2-4k = 0 --> c=sqrt(4k)
# CBF underdamped for undershoot --> c = 0.4*sqrt(4k)
# CMRO2 critical --> c = sqrt(4k)
# after CBF and CMRO2 standard balloon model with revised coefficients, parameter values = Friston et al. (2000)
BoldNeuron_new = Neuron(
parameters = """
    c_CBF       = 1
    k_CBF       = 1
    c_CMRO2     = 1
    k_CMRO2     = 1
    ea          = 0.005
    E_0         = 0.34
    tau_0       = 0.98
    alpha       = 0.33
    V_0         = 0.02
    v_0         = 40.3
    TE          = 40/1000.
    epsilon     = 1
    r_0         = 25
    M_Davis     = 11.1
    alpha_Davis = 0.38
    beta_Davis  = 1.5
""",
equations = """
    I_CBF           = sum(I_ampa) + 1.5 * sum(I_gaba)                              : init=0
    I_CMRO2         = sum(I_ampa) * (k_CMRO2 / k_CBF)                              : init=0
    1000*dsCBF/dt   = ea * I_CBF - c_CBF * sCBF - k_CBF * (CBF - 1)                : init=0
    1000*dCBF/dt    = sCBF                                                         : init=1, max=2, min=0
    1000*dsCMRO2/dt = ea * I_CMRO2 - c_CMRO2 * sCMRO2 - k_CMRO2 * (CMRO2 - 1)      : init=0
    1000*dCMRO2/dt  = sCMRO2                                                       : init=1, max=2, min=0

    1000*dq/dt      = 1 / tau_0 * (CMRO2 - (q / v) * f_out)                        : init=1
    1000*dv/dt      = 1 / tau_0 * (CBF - f_out)                                    : init=1
    f_out           = v**(1 / alpha)                                               : init=1

    k_1             = 4.3 * v_0 * E_0 * TE
    k_2             = epsilon * r_0 * E_0 * TE
    k_3             = 1 - epsilon
    BOLD_Balloon    = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
    BOLD_Davis      = M_Davis * (1 - CBF**alpha_Davis * (CMRO2 / CBF)**beta_Davis) : init=0
    r=0
""",
    name = "-",
    description = "-"
)