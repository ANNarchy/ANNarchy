#===============================================================================
#
#     PredefinedModels.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Oliver Maith <oli_maith@gmx.de>,
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
from .BoldModel import BoldModel

#our current BOLD computation
balloon_maith2021 = BoldModel(
parameters = """
    phi_CBF    = 1.0
    kappa_CBF = 0.665
    gamma_CBF = 0.412
    E_0   = 0.3424
    tau_0 = 1.0368
    alpha = 0.3215
    V_0   = 0.02
""",
equations = """
    I_CBF          = sum(exc)                                                     : init=0
    1000*ds_CBF/dt = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    1000*df_in/dt  = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    1000*dq/dt     = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    1000*dv/dt     = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 7 * E_0
    k_2            = 2
    k_3            = 2 * E_0 - 0.2
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0

    r=0
""",
    name = "BoldNeuron",
    description = "BOLD computation from Maith et al. (2021)."
)


#Stephan et al. (2007)
#V_0, tau_0, tau_s, tau_f they don't mention --> use Frsiton et al. 2000
#E_0, epsilon are free --> E_0 use Friston et al. 2000, epsilon use 1.43 (probably used if 1.43 is fixed in Stephan et al. (2007), value from Obata et al. (2004))
balloon_CN = BoldModel(
parameters = """
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000.
    epsilon   = 1.43
""",
equations = """
    I_CBF          = sum(exc)                                                     : init=0
    1000*ds_CBF/dt = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    1000*df_in/dt  = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    1000*dq/dt     = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    1000*dv/dt     = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2            = 2 * E_0
    k_3            = 1 - epsilon
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
""",
    name = "BoldNeuron CBN",
    description = "BOLD computation with classic coefficients and non-linear BOLD equation (Stephan et al., 2007)."
)


balloon_CL = BoldModel(
parameters = """
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000.
    epsilon   = 1.43
""",
equations = """
    I_CBF          = sum(exc)                                                     : init=0
    1000*ds_CBF/dt = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    1000*df_in/dt  = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    1000*dq/dt     = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    1000*dv/dt     = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2            = 2 * E_0
    k_3            = 1 - epsilon
    BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))        : init=0
""",
    name = "BoldNeuron CBL",
    description = "BOLD computation with classic coefficients and linear BOLD equation (Stephan et al., 2007)."
)


balloon_RN = BoldModel(
parameters = """
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000.
    epsilon   = 1.43
    r_0       = 25
""",
equations = """
    I_CBF          = sum(exc)                                                     : init=0
    1000*ds_CBF/dt = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    1000*df_in/dt  = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    1000*dq/dt     = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    1000*dv/dt     = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 4.3 * v_0 * E_0 * TE
    k_2            = epsilon * r_0 * E_0 * TE
    k_3            = 1 - epsilon
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
""",
    name = "BoldNeuron RBN",
    description = "BOLD computation with revised coefficients and non-linear BOLD equation (Stephan et al., 2007)."
)


balloon_RL = BoldModel(
parameters = """
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000.
    epsilon   = 1.43
    r_0       = 25
""",
equations = """
    I_CBF          = sum(exc)                                                     : init=0
    1000*ds_CBF/dt = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    1000*df_in/dt  = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    1000*dq/dt     = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    1000*dv/dt     = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 4.3 * v_0 * E_0 * TE
    k_2            = epsilon * r_0 * E_0 * TE
    k_3            = 1 - epsilon
    BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))        : init=0
""",
    name = "BoldNeuron RBL",
    description = "BOLD computation with revised coefficients and linear BOLD equation (Stephan et al., 2007)."
)


# new model
# damped harmonic oscillators, k->timeconstant, c->damping
# CBF --> gamma from Friston
# CMRO2 --> faster --> gamma=gamma_CBF*10 (therefore scaling of I_CMRO2 by (gamma_CMRO2 / gamma_CBF) --> if same input (I_CBF==I_CMRO2) CMRO2 and CBF same steady-state)
# critical kappa --> kappa**2-4*gamma = 0 --> kappa=sqrt(4*gamma)
# CBF underdamped for undershoot --> kappa = 0.6*sqrt(4*gamma)
# CMRO2 critical --> kappa = sqrt(4*gamma)
# after CBF and CMRO2 standard balloon model with revised coefficients, parameter values = Friston et al. (2000)
balloon_two_inputs = BoldModel(
parameters = """
    kappa_CBF   = 0.7650920556760059
    gamma_CBF   = 1/2.46
    kappa_CMRO2 = 4.032389192727559
    gamma_CMRO2 = 10/2.46
    phi_CBF     = 1.0
    phi_CMRO2   = 1.0
    E_0         = 0.34
    tau_0       = 0.98
    alpha       = 0.33
    V_0         = 0.02
    v_0         = 40.3
    TE          = 40/1000.
    epsilon     = 1
    r_0         = 25
""",
equations = """
    I_CBF            = sum(I_f)                                                                                        : init=0
    1000*ds_CBF/dt   = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1)                                    : init=0
    1000*df_in/dt    = s_CBF                                                                                           : init=1, min=0.01

    I_CMRO2          = sum(I_r)                                                                                        : init=0
    1000*ds_CMRO2/dt = phi_CMRO2 * I_CMRO2 * (gamma_CMRO2 / gamma_CBF) - kappa_CMRO2 * s_CMRO2 - gamma_CMRO2 * (r - 1) : init=0
    1000*dr/dt       = s_CMRO2                                                                                         : init=1, min=0.01

    1000*dq/dt       = 1 / tau_0 * (r - (q / v) * f_out)                                                               : init=1, min=0.01
    1000*dv/dt       = 1 / tau_0 * (f_in - f_out)                                                                      : init=1, min=0.01
    f_out            = v**(1 / alpha)                                                                                  : init=1, min=0.01

    k_1              = 4.3 * v_0 * E_0 * TE
    k_2              = epsilon * r_0 * E_0 * TE
    k_3              = 1 - epsilon
    BOLD             = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))                                     : init=0
""",
    name = "new BOLD model",
    description = "BOLD model with two inputs (CBF-driving and CMRO2-driving). Combination of neurovascular coupling of Friston et al. (2000) and non-linear Balloon model with revised coefficients (Buxton et al, 1998, Stephan et al, 2007)"
)
