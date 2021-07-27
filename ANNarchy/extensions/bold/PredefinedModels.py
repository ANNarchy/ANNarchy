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

#################################
### Our current BOLD computation
#################################
class balloon_maith2021(BoldModel):
    """
    The balloon model as used in:

    >  Maith et al. (2021) A computational model-based analysis of basal ganglia pathway changes in Parkinson’s disease inferred from resting-state fMRI. European Journal of Neuroscience. 2021; 53: 2278– 2295. https://doi.org/10.1111/ejn.14868 

    Please note:

    - the model expects an input_variable = "exc".
    """
    def __init__(self):
        parameters = """
    tau       = 1000.0
    phi_CBF   = 1.0
    kappa_CBF = 0.665
    gamma_CBF = 0.412
    E_0       = 0.3424
    tau_0     = 1.0368
    alpha     = 0.3215
    V_0       = 0.02
"""
        equations = """
    I_CBF          = sum(exc)                                                     : init=0
    tau*ds_CBF/dt  = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    tau*df_in/dt   = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    tau*dq/dt      = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    tau*dv/dt      = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 7 * E_0
    k_2            = 2
    k_3            = 2 * E_0 - 0.2
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
"""
        name = "BoldNeuron"
        description = "BOLD computation from Maith et al. (2021)."

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)

#####################################################################
### Several balloon model variants following Stephan et al. (2007)
###
### C = classical coefficient (e. g. no epsilon influence on k_2)
### R = revised coefficient (e. g. epsilon influence on k_2)
###
### N = non-linear BOLD equation
### L = linear bold equation
#####################################################################
class balloon_CN(BoldModel):
    """
    A balloon model with classic coefficient and non-linear BOLD equation derived from:

    Stephan et al. (2007). Comparing hemodynamic models with dcm. Neuroimage 38, 387–401
    """
    def __init__(self):
        # Model comments:
        #   V_0, tau_0, tau_s, tau_f they don't mention --> use Frsiton et al. 2000
        #   E_0, epsilon are free --> E_0 use Friston et al. 2000, epsilon use 1.43 (probably used if 1.43 is fixed in Stephan et al. (2007), value from Obata et al. (2004))
        parameters = """
    tau       = 1000.0
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000. # TODO: 40./tau ?
    epsilon   = 1.43
"""
        equations = """
    I_CBF          = sum(exc)                                                     : init=0
    tau*ds_CBF/dt  = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    tau*df_in/dt   = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    tau*dq/dt      = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    tau*dv/dt      = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2            = 2 * E_0
    k_3            = 1 - epsilon
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
"""
        name = "BoldNeuron CN",
        description = "BOLD computation with classic coefficients and non-linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)


class balloon_CL(BoldModel):
    """
    A balloon model with classical coefficient and linear BOLD equation derived from:

    Stephan et al. (2007). Comparing hemodynamic models with dcm. Neuroimage 38, 387–401
    """
    def __init__(self):
        parameters = """
    tau       = 1000.0
    phi_CBF   = 1.0
    kappa_CBF = 1/1.54
    gamma_CBF = 1/2.46
    E_0       = 0.34
    tau_0     = 0.98
    alpha     = 0.33
    V_0       = 0.02
    v_0       = 40.3
    TE        = 40/1000.    # TODO: 40./tau ?
    epsilon   = 1.43
"""
        equations = """
    I_CBF          = sum(exc)                                                     : init=0
    tau*ds_CBF/dt  = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    tau*df_in/dt   = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    tau*dq/dt      = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    tau*dv/dt      = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
    k_2            = 2 * E_0
    k_3            = 1 - epsilon
    BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))        : init=0
"""
        name = "BoldNeuron CL"
        description = "BOLD computation with classic coefficients and linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)

class balloon_RN(BoldModel):
    """
    A balloon model with revised coefficient and non-linear BOLD equation derived from:

    Stephan et al. (2007). Comparing hemodynamic models with dcm. Neuroimage 38, 387–401
    """
    def __init__(self):
        parameters = """
    tau       = 1000.0
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
"""
        equations = """
    I_CBF          = sum(exc)                                                     : init=0
    tau*ds_CBF/dt  = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    tau*df_in/dt   = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    tau*dq/dt      = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    tau*dv/dt      = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 4.3 * v_0 * E_0 * TE
    k_2            = epsilon * r_0 * E_0 * TE
    k_3            = 1 - epsilon
    BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
"""
        name = "BoldNeuron RN"
        description = "BOLD computation with revised coefficients and non-linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)


class balloon_RL(BoldModel):
    """
    A balloon model with revised coefficients and linear BOLD equation derived from:

    Stephan et al. (2007). Comparing hemodynamic models with dcm. Neuroimage 38, 387–401
    """
    def __init__(self):
        parameters = """
    tau       = 1000.0
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
"""
        equations = """
    I_CBF          = sum(exc)                                                     : init=0
    tau*ds_CBF/dt  = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
    tau*df_in/dt   = s_CBF                                                        : init=1, min=0.01

    E              = 1 - (1 - E_0)**(1 / f_in)                                    : init=0.3424
    tau*dq/dt      = 1 / tau_0 * (f_in * E / E_0 - (q / v) * f_out)               : init=1, min=0.01
    tau*dv/dt      = 1 / tau_0 * (f_in - f_out)                                   : init=1, min=0.01
    f_out          = v**(1 / alpha)                                               : init=1, min=0.01

    k_1            = 4.3 * v_0 * E_0 * TE
    k_2            = epsilon * r_0 * E_0 * TE
    k_3            = 1 - epsilon
    BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))        : init=0
"""
        name = "BoldNeuron RL"
        description = "BOLD computation with revised coefficients and linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)

#################################
### Two input channel model
#################################
class balloon_two_inputs(BoldModel):
    """
    Implement a model with two input signals (CBF-driving and CMRO2-driving) for the ballon model and non-linear BOLD equation.
    """
    def __init__(self):
        # damped harmonic oscillators, k->timeconstant, c->damping
        # CBF --> gamma from Friston
        # CMRO2 --> faster --> gamma=gamma_CBF*10 (therefore scaling of I_CMRO2 by (gamma_CMRO2 / gamma_CBF) --> if same input (I_CBF==I_CMRO2) CMRO2 and CBF same steady-state)
        # critical kappa --> kappa**2-4*gamma = 0 --> kappa=sqrt(4*gamma)
        # CBF underdamped for undershoot --> kappa = 0.6*sqrt(4*gamma)
        # CMRO2 critical --> kappa = sqrt(4*gamma)
        # after CBF and CMRO2 standard balloon model with revised coefficients, parameter values = Friston et al. (2000)
        parameters = """
    tau         = 1000.0
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
"""
        equations = """
    I_CBF            = sum(I_f)                                                                                        : init=0
    tau*ds_CBF/dt    = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1)                                    : init=0
    tau*df_in/dt     = s_CBF                                                                                           : init=1, min=0.01

    I_CMRO2          = sum(I_r)                                                                                        : init=0
    tau*ds_CMRO2/dt  = phi_CMRO2 * I_CMRO2 * (gamma_CMRO2 / gamma_CBF) - kappa_CMRO2 * s_CMRO2 - gamma_CMRO2 * (r - 1) : init=0
    tau*dr/dt        = s_CMRO2                                                                                         : init=1, min=0.01

    tau*dq/dt        = 1 / tau_0 * (r - (q / v) * f_out)                                                               : init=1, min=0.01
    tau*dv/dt        = 1 / tau_0 * (f_in - f_out)                                                                      : init=1, min=0.01
    f_out            = v**(1 / alpha)                                                                                  : init=1, min=0.01

    k_1              = 4.3 * v_0 * E_0 * TE
    k_2              = epsilon * r_0 * E_0 * TE
    k_3              = 1 - epsilon
    BOLD             = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))                                     : init=0
"""
        name = "new BOLD model"
        description = "BOLD model with two inputs (CBF-driving and CMRO2-driving). Combination of neurovascular coupling of Friston et al. (2000) and non-linear Balloon model with revised coefficients (Buxton et al, 1998, Stephan et al, 2007)"

        BoldModel.__init__(self, parameters=parameters, equations=equations, name=name, description=description)
