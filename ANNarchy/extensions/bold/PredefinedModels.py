"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from .BoldModel import BoldModel

#####################################################################
### Several balloon model variants following Stephan et al. (2007)
###
### C = classical coefficient (e. g. no epsilon influence on k_2)
### R = revised coefficient (e. g. epsilon influence on k_2)
###
### N = non-linear BOLD equation
### L = linear bold equation
###
### Parameters:
### V_0, tau_0, tau_s, tau_f they don't mention --> use Friston et al. (2000)
### E_0, epsilon are free --> E_0 use Friston et al. (2000), epsilon use 1.43 (probably used if 1.43 is fixed in Stephan et al. (2007), value from Obata et al. (2004))
###
### References:
### Friston et al. (2000). Nonlinear responses in fmri: the balloon model, volterra kernels, and other hemodynamics.NeuroImage12, 466–477
### Obata et al. (2004).  Discrepancies between bold and flow dynamics in primary and supplementary motor areas: application of the balloon model to the interpretation of bold transients. NeuroImage 21, 144–153
#####################################################################
class balloon_CN(BoldModel):
    """
    A balloon model with classic coefficients and non-linear BOLD equation derived from Stephan et al. (2007).

    Equivalent code:

    ```python
    balloon_CN = BoldModel(
        parameters = dict(
            second    = 1000.0,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
        ),
        equations = [
            # Single input
            ann.Variable('I_CBF = sum(I_CBF)', init=0.0),
            ann.Variable('ds/dt = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second', init=0.0),
            ann.Variable('df_in/dt = s / second', init=1.0, min=0.01),

            ann.Variable('E = 1 - (1 - E_0)**(1 / f_in)', init=0.3424),
            ann.Variable('dq/dt = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)', init=1.0, min=0.01),
            ann.Variable('dv/dt = (f_in - f_out)/(tau*second), init=1.0, min=0.01),
            ann.Variable('f_out = v**(1 / alpha)', init=1, min=0.01),

            # Classic coefficients
            ann.Variable('k_1 = (1 - V_0) * 4.3 * v_0 * E_0 * TE'),
            ann.Variable('k_2 = 2 * E_0'),
            ann.Variable('k_3 = 1 - epsilon'),

            # Non-linear equation
            ann.Variable('BOLD = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))'),
        ],
        inputs="I_CBF",
    )
    ```

    :param phi:       input coefficient
    :param kappa:     signal decay
    :param gamma:     feedback regulation
    :param E_0:       oxygen extraction fraction at rest
    :param tau:       time constant (in s!)
    :param alpha:     vessel stiffness
    :param V_0:       resting venous blood volume fraction
    :param v_0:       frequency offset at the outer surface of the magnetized vessel for fully deoxygenated blood at 1.5 T
    :param TE:        echo time
    :param epsilon:   ratio of intra- and extravascular signal
    """
    def __init__(self,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
        ):

        parameters = f"""
            second    = 1000.0 : population
            phi       = {phi} : population
            kappa     = {kappa} : population
            gamma     = {gamma} : population
            E_0       = {E_0} : population
            tau       = {tau} : population
            alpha     = {alpha} : population
            V_0       = {V_0} : population
            v_0       = {v_0} : population
            TE        = {TE} : population
            epsilon   = {epsilon} : population
        """ 

        equations = """
            # Single input
            I_CBF          = sum(I_CBF)                                                : init=0
            ds/dt          = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second     : init=0
            df_in/dt       = s / second                                                : init=1, min=0.01

            E              = 1 - (1 - E_0)**(1 / f_in)                                 : init=0.3424
            dq/dt          = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)           : init=1, min=0.01
            dv/dt          = (f_in - f_out)/(tau*second)                               : init=1, min=0.01
            f_out          = v**(1 / alpha)                                            : init=1, min=0.01

            # Classic coefficients
            k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
            k_2            = 2 * E_0
            k_3            = 1 - epsilon

            # Non-linear equation
            BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
        """
        name = "BoldNeuron CN",
        description = "BOLD computation with classic coefficients and non-linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations,  
            inputs="I_CBF",
            output="BOLD",
            name=name, description=description
        )
        

class balloon_CL(BoldModel):
    """
    A balloon model with classical coefficients and linear BOLD equation derived from Stephan et al. (2007).

    Equivalent code:

    ```python
    balloon_CL = BoldModel(
        parameters = dict(
            second    = 1000.0,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
        ),
        equations = [
            # Single input
            ann.Variable('I_CBF = sum(I_CBF)', init=0.0),
            ann.Variable('ds/dt = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second', init=0.0),
            ann.Variable('df_in/dt = s / second', init=1.0, min=0.01),

            ann.Variable('E = 1 - (1 - E_0)**(1 / f_in)', init=0.3424),
            ann.Variable('dq/dt = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)', init=1.0, min=0.01),
            ann.Variable('dv/dt = (f_in - f_out)/(tau*second), init=1.0, min=0.01),
            ann.Variable('f_out = v**(1 / alpha)', init=1, min=0.01),

            # Classic coefficients
            ann.Variable('k_1 = (1 - V_0) * 4.3 * v_0 * E_0 * TE'),
            ann.Variable('k_2 = 2 * E_0'),
            ann.Variable('k_3 = 1 - epsilon'),

            # Non-linear equation
            ann.Variable('BOLD = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))'),
        ],
        inputs="I_CBF",
    )
    ```

    :param phi:       input coefficient
    :param kappa:     signal decay
    :param gamma:     feedback regulation
    :param E_0:       oxygen extraction fraction at rest
    :param tau:       time constant (in s!)
    :param alpha:     vessel stiffness
    :param V_0:       resting venous blood volume fraction
    :param v_0:       frequency offset at the outer surface of the magnetized vessel for fully deoxygenated blood at 1.5 T
    :param TE:        echo time
    :param epsilon:   ratio of intra- and extravascular signal
    """
    def __init__(self,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
        ):

        parameters = f"""
            second    = 1000.0 : population
            phi       = {phi} : population
            kappa     = {kappa} : population
            gamma     = {gamma} : population
            E_0       = {E_0} : population
            tau       = {tau} : population
            alpha     = {alpha} : population
            V_0       = {V_0} : population
            v_0       = {v_0} : population
            TE        = {TE} : population
            epsilon   = {epsilon} : population
        """ 

        equations = """
            # Single input
            I_CBF          = sum(I_CBF)                                                : init=0
            ds/dt          = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second     : init=0
            df_in/dt       = s / second                                                : init=1, min=0.01

            E              = 1 - (1 - E_0)**(1 / f_in)                                 : init=0.3424
            dq/dt          = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)           : init=1, min=0.01
            dv/dt          = (f_in - f_out)/(tau*second)                               : init=1, min=0.01
            f_out          = v**(1 / alpha)                                            : init=1, min=0.01

            # Classic coefficients
            k_1            = (1 - V_0) * 4.3 * v_0 * E_0 * TE
            k_2            = 2 * E_0
            k_3            = 1 - epsilon

            # Linear equation
            BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))        : init=0
        """
        name = "BoldNeuron CL"
        description = "BOLD computation with classic coefficients and linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations,  
            inputs="I_CBF",
            output="BOLD",
            name=name, description=description
        )

class balloon_RN(BoldModel):
    """
    A balloon model with revised coefficients and non-linear BOLD equation derived from Stephan et al. (2007).

    Equivalent code:

    ```python
    balloon_RN = BoldModel(
        parameters = dict(
            second    = 1000.0,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
            r_0       = 25.,
        ),
        equations = [
            # Single input
            ann.Variable('I_CBF = sum(I_CBF)', init=0.0),
            ann.Variable('ds/dt = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second', init=0.0),
            ann.Variable('df_in/dt = s / second', init=1.0, min=0.01),

            ann.Variable('E = 1 - (1 - E_0)**(1 / f_in)', init=0.3424),
            ann.Variable('dq/dt = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)', init=1.0, min=0.01),
            ann.Variable('dv/dt = (f_in - f_out)/(tau*second), init=1.0, min=0.01),
            ann.Variable('f_out = v**(1 / alpha)', init=1, min=0.01),

            # Revised coefficients
            ann.Variable('k_1 = 4.3 * v_0 * E_0 * TE'),
            ann.Variable('k_2 = epsilon * r_0 * E_0 * TE'),
            ann.Variable('k_3 = 1 - epsilon'),

            # Non-linear equation
            ann.Variable('BOLD = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))'),
        ],
        inputs="I_CBF",
    )
    ```

    :param phi:       input coefficient
    :param kappa:     signal decay
    :param gamma:     feedback regulation
    :param E_0:       oxygen extraction fraction at rest
    :param tau:       time constant (in s!)
    :param alpha:     vessel stiffness
    :param V_0:       resting venous blood volume fraction
    :param v_0:       frequency offset at the outer surface of the magnetized vessel for fully deoxygenated blood at 1.5 T
    :param TE:        echo time
    :param epsilon:   ratio of intra- and extravascular signal
    :param r_0:       slope of the relation between the intravascular relaxation rate and oxygen saturation
    """
    def __init__(self,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
            r_0       = 25,
        ):

        parameters = f"""
            second    = 1000.0 : population
            phi       = {phi} : population
            kappa     = {kappa} : population
            gamma     = {gamma} : population
            E_0       = {E_0} : population
            tau       = {tau} : population
            alpha     = {alpha} : population
            V_0       = {V_0} : population
            v_0       = {v_0} : population
            TE        = {TE} : population
            epsilon   = {epsilon} : population
            r_0       = {r_0} : population
        """

        equations = """
            # Single input
            I_CBF          = sum(I_CBF)                                                : init=0
            ds/dt          = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second     : init=0
            df_in/dt       = s / second                                                : init=1, min=0.01

            E              = 1 - (1 - E_0)**(1 / f_in)                                 : init=0.3424
            dq/dt          = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)           : init=1, min=0.01
            dv/dt          = (f_in - f_out)/(tau*second)                               : init=1, min=0.01
            f_out          = v**(1 / alpha)                                            : init=1, min=0.01

            # Revised coefficients
            k_1            = 4.3 * v_0 * E_0 * TE
            k_2            = epsilon * r_0 * E_0 * TE
            k_3            = 1.0 - epsilon

            # Non-linear equation
            BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
        """
        name = "BOLD model RN"
        description = "BOLD computation with revised coefficients and non-linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations,  
            inputs="I_CBF",
            output="BOLD",
            name=name, description=description
        )

class balloon_RL(BoldModel):
    """
    A balloon model with revised coefficients and linear BOLD equation derived from Stephan et al. (2007).

    Equivalent code:

    ```python
    balloon_RL = BoldModel(
        parameters = dict(
            second    = 1000.0,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
            r_0       = 25.,
        ),
        equations = [
            # Single input
            ann.Variable('I_CBF = sum(I_CBF)', init=0.0),
            ann.Variable('ds/dt = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second', init=0.0),
            ann.Variable('df_in/dt = s / second', init=1.0, min=0.01),

            ann.Variable('E = 1 - (1 - E_0)**(1 / f_in)', init=0.3424),
            ann.Variable('dq/dt = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)', init=1.0, min=0.01),
            ann.Variable('dv/dt = (f_in - f_out)/(tau*second), init=1.0, min=0.01),
            ann.Variable('f_out = v**(1 / alpha)', init=1, min=0.01),

            # Revised coefficients
            ann.Variable('k_1 = 4.3 * v_0 * E_0 * TE'),
            ann.Variable('k_2 = epsilon * r_0 * E_0 * TE'),
            ann.Variable('k_3 = 1 - epsilon'),

            # Linear equation
            ann.Variable('BOLD = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))'),
        ],
        inputs="I_CBF",
    )
    ```

    :param phi:       input coefficient
    :param kappa:     signal decay
    :param gamma:     feedback regulation
    :param E_0:       oxygen extraction fraction at rest
    :param tau:       time constant (in s!)
    :param alpha:     vessel stiffness
    :param V_0:       resting venous blood volume fraction
    :param v_0:       frequency offset at the outer surface of the magnetized vessel for fully deoxygenated blood at 1.5 T
    :param TE:        echo time
    :param epsilon:   ratio of intra- and extravascular signal
    :param r_0:       slope of the relation between the intravascular relaxation rate and oxygen saturation

    """
    def __init__(self,
            phi       = 1.0,
            kappa     = 1/1.54,
            gamma     = 1/2.46,
            E_0       = 0.34,
            tau       = 0.98,
            alpha     = 0.33,
            V_0       = 0.02,
            v_0       = 40.3,
            TE        = 40/1000.,
            epsilon   = 1.43,
            r_0       = 25,
        ):

        parameters = f"""
            second    = 1000.0 : population
            phi       = {phi} : population
            kappa     = {kappa} : population
            gamma     = {gamma} : population
            E_0       = {E_0} : population
            tau       = {tau} : population
            alpha     = {alpha} : population
            V_0       = {V_0} : population
            v_0       = {v_0} : population
            TE        = {TE} : population
            epsilon   = {epsilon} : population
            r_0       = {r_0} : population
        """

        equations = """
            # Single input
            I_CBF          = sum(I_CBF)                                                    : init=0
            ds/dt          = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second         : init=0
            df_in/dt       = s  / second                                                   : init=1, min=0.01

            E              = 1 - (1 - E_0)**(1 / f_in)                                     : init=0.3424
            tau*dq/dt      = f_in * E / E_0 - (q / v) * f_out                              : init=1, min=0.01
            tau*dv/dt      = f_in - f_out                                                  : init=1, min=0.01
            f_out          = v**(1 / alpha)                                                : init=1, min=0.01

            # Revised coefficients
            k_1            = 4.3 * v_0 * E_0 * TE
            k_2            = epsilon * r_0 * E_0 * TE
            k_3            = 1 - epsilon

            # Linear equation
            BOLD           = V_0 * ((k_1 + k_2) * (1 - q) + (k_3 - k_2) * (1 - v))         : init=0
        """
        name = "BOLD model RL"
        description = "BOLD computation with revised coefficients and linear BOLD equation (Stephan et al., 2007)."

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations,  
            inputs="I_CBF",
            output="BOLD",
            name=name, description=description
        )

#################################
### Two input channel model
#################################
class balloon_two_inputs(BoldModel):
    """
    BOLD model with two input signals (CBF-driving and CMRO2-driving) for the ballon model and non-linear BOLD equation with revised coefficients based on Buxton et al. (2004), Friston et al. (2000) and Stephan et al. (2007).
    """
    def __init__(self):

        # damped harmonic oscillators, gamma->spring coefficient, kappa->damping coefficient
        # CBF --> gamma from Friston
        # CMRO2 --> faster --> gamma=gamma_CBF*10 (therefore scaling of I_CMRO2 by (gamma_CMRO2 / gamma_CBF) --> if same input (I_CBF==I_CMRO2) CMRO2 and CBF same steady-state)
        # critical kappa --> kappa**2-4*gamma = 0 --> kappa=sqrt(4*gamma)
        # CBF underdamped for undershoot --> kappa = 0.6*sqrt(4*gamma)
        # CMRO2 critical --> kappa = sqrt(4*gamma)
        # after CBF and CMRO2 standard balloon model with revised coefficients, parameter values = Friston et al. (2000)
        parameters = """
            kappa_CBF   = 0.7650920556760059
            gamma_CBF   = 1/2.46
            kappa_CMRO2 = 4.032389192727559
            gamma_CMRO2 = 10/2.46
            phi_CBF     = 1.0
            phi_CMRO2   = 1.0
            E_0         = 0.34
            tau         = 0.98
            alpha       = 0.33
            V_0         = 0.02
            v_0         = 40.3
            TE          = 40/1000.
            epsilon     = 1
            r_0         = 25
            tau_out1    = 0
            tau_out2    = 20
            second      = 1000
        """
        equations = """
            # CBF input
            I_CBF               = sum(I_CBF)                                                   : init=0
            second*ds_CBF/dt    = phi_CBF * I_CBF - kappa_CBF * s_CBF - gamma_CBF * (f_in - 1) : init=0
            second*df_in/dt     = s_CBF                                                        : init=1, min=0.01

            # CMRO2 input
            I_CMRO2             = sum(I_CMRO2)                                                 : init=0
            second*ds_CMRO2/dt  = phi_CMRO2 * I_CMRO2 * (gamma_CMRO2 / gamma_CBF) - kappa_CMRO2 * s_CMRO2 - gamma_CMRO2 * (r - 1) : init=0
            second*dr/dt        = s_CMRO2                                                      : init=1, min=0.01

            dv                  = f_in - v**(1 / alpha)
            tau_out             = if dv>0: tau_out1 else: tau_out2
            f_out               = v**(1/alpha) + tau_out * dv / (tau + tau_out)              : init=1, min=0.01
            
            dq/dt               = (r - (q / v) * f_out)  / (second*tau)                        : init=1, min=0.01
            dv/dt               = dv / (tau + tau_out)  / second                             : init=1, min=0.01

            # Revised coefficients
            k_1                 = 4.3 * v_0 * E_0 * TE
            k_2                 = epsilon * r_0 * E_0 * TE
            k_3                 = 1 - epsilon

            # Non-linear equation
            BOLD                = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v))  : init=0
        """
        name = "BOLD model with two inputs"
        description = "BOLD model with two inputs (CBF-driving and CMRO2-driving). Combination of neurovascular coupling of Friston et al. (2000) and non-linear Balloon model with revised coefficients (Buxton et al, 1998, Stephan et al, 2007)"

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations, 
            inputs=['I_CBF', 'I_CMRO2'],
            output="BOLD",
            name=name, 
            description=description
        )


#################################
### Our current BOLD computation
#################################
class balloon_maith2021(BoldModel):
    """
    The balloon model as used in Maith et al. (2021).
    """
    def __init__(self):

        parameters = """
            second    = 1000.0
            phi   = 1.0
            kappa = 0.665
            gamma = 0.412
            E_0       = 0.3424
            tau     = 1.0368
            alpha     = 0.3215
            V_0       = 0.02
        """
        equations = """
            I_CBF          = sum(I_CBF)                                                 : init=0
            ds/dt          = (phi * I_CBF - kappa * s - gamma * (f_in - 1))/second      : init=0
            df_in/dt       = s / second                                                 : init=1, min=0.01

            E              = 1 - (1 - E_0)**(1 / f_in)                                  : init=0.3424
            dq/dt          = (f_in * E / E_0 - (q / v) * f_out)/(tau*second)            : init=1, min=0.01
            dv/dt          = (f_in - f_out)/(tau*second)                                : init=1, min=0.01
            f_out          = v**(1 / alpha)                                             : init=1, min=0.01

            k_1            = 7 * E_0
            k_2            = 2
            k_3            = 2 * E_0 - 0.2

            BOLD           = V_0 * (k_1 * (1 - q) + k_2 * (1 - (q / v)) + k_3 * (1 - v)) : init=0
        """
        name = "Maith2021 BOLD model"
        description = "BOLD computation from Maith et al. (2021)."

        BoldModel.__init__(self, 
            parameters=parameters, 
            equations=equations,  
            inputs="I_CBF",
            output="BOLD",
            name=name, description=description
        )