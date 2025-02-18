"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Neuron import Neuron

class BoldModel(Neuron):
    r"""
    Base class to define a BOLD model to be used in a BOLD monitor.

    A BOLD model is quite similar to a regular rate-coded neuron. It gets a weighted sum of inputs with a specific target (e.g. I_CBF) and compute a single output variable (called `BOLD` in the predefined models, but it could be anything).

    The main difference is that a BOLD model should also declare which targets are used for the input signal:

    ```python
    bold_model = BoldModel(
        parameters = dict(
            tau = 1000.
        ),
        equations = [
            'I_CBF = sum(I_CBF)',
            # ...
            'tau * dBOLD/dt = I_CBF - BOLD',
        ],
        inputs = "I_CBF"
    )
    ```

    The provided BOLD models follow the Balloon model (Buxton et al., 1998) with the different variations studied in (Stephan et al., 2007). Those models all compute the vascular response to neural activity through a dampened oscillator:

    $$
        \frac{ds}{dt} = \phi \, I_\text{CBF} - \kappa \, s - \gamma \, (f_{in} - 1)
    $$

    $$
        \frac{df_{in}}{dt} = s
    $$

    This allows to compute the oxygen extraction fraction:

    $$
        E = 1 - (1 - E_{0})^{ \frac{1}{f_{in}} }
    $$

    The (normalized) venous blood volume is computed as:

    $$
        \tau_0 \, \frac{dv}{dt} = (f_{in} - f_{out})
    $$

    $$
        f_{out} = v^{\frac{1}{\alpha}}
    $$

    The level of deoxyhemoglobin into the venous compartment is computed by:

    $$
        \tau_0 \, \frac{dq}{dt} = f_{in} \, \frac{E}{E_0} - \frac{q}{v} \, f_{out}
    $$

    Using the two signals $v$ and $q$, there are two ways to compute the corresponding BOLD signal:

    * **N:** Non-linear BOLD equation:

    $$
        BOLD = v_0 \, ( k_1 \, (1-q) + k_2 \, (1- \dfrac{q}{v}) + k_3 \, (1 - v) )
    $$

    * **L:** Linear BOLD equation:

    $$
        BOLD = v_0 \, ((k_1 + k_2) \, (1 - q) + (k_3 - k_2) \, (1 - v)) 
    $$

    Additionally, the three coefficients $k_1$, $k_2$, $k_3$ can be computed in two different ways:

    * **C:** classical coefficients from (Buxton et al., 1998):

    $$k_1            = (1 - v_0) \, 4.3 \, v_0 \, E_0 \, \text{TE}$$

    $$k_2            = 2 \, E_0$$

    $$k_3            = 1 - \epsilon$$

    * **R:** revised coefficients from (Obata et al., 2004):

    $$k_1            = 4.3 \, v_0 \, E_0 \, \text{TE}$$

    $$k_2            = \epsilon \, r_0 \, E_0 \, \text{TE}$$

    $$k_3            = 1 - \epsilon$$

    This makes a total of four different BOLD model (`balloon_RN`, `balloon_RL`, `balloon_CN`, `balloon_CL`) which are provided by the extension. The different parameters can be modified in the constructor. Additionally, we also provide the model that was used in (Maith et al., 2021) and the two-inputs model of (Maith et al, 2022).
    
    > Buxton, R. B., Wong, E. C., and Frank, L. R. (1998). Dynamics of blood flow and oxygenation changes during brain activation: the balloon model. Magnetic resonance in medicine 39, 855-864. doi:10.1002/mrm.1910390602
    
    > Friston et al. (2000). Nonlinear responses in fMRI: the balloon model, volterra kernels, and other hemodynamics. NeuroImage 12, 466-477
    
    > Buxton et al. (2004). Modeling the hemodynamic response to brain activation. Neuroimage 23, 220-233. doi:10.1016/j.neuroimage.2004.07.013
    
    > Stephan et al. (2007). Comparing hemodynamic models with DCM. Neuroimage 38, 387-401. doi:10.1016/j.neuroimage.2007.07.040
    
    > Maith et al. (2021) A computational model-based analysis of basal ganglia pathway changes in Parkinson's disease inferred from resting-state fMRI. European Journal of Neuroscience. 2021; 53: 2278-2295. doi:10.1111/ejn.14868 
    
    > Maith et al. (2022) BOLD monitoring in the neural simulator ANNarchy. Frontiers in Neuroinformatics 16. doi:10.3389/fninf.2022.790966.

    :param parameters: parameters of the model and their initial value.
    :param equations: equations defining the temporal evolution of variables.
    :param inputs: single variable or list of input signals (e.g. 'I_CBF' or ['I_CBF', 'I_CMRO2']).
    :param output: output variable of the model (default is 'BOLD').
    :param name: optional model name.
    :param description: optional model description.
    """
    def __init__(self, parameters, equations, inputs, output=["BOLD"], name="Custom BOLD model", description=""):

        # The processing in BoldMonitor expects lists, but the interface
        # should allow also single strings (if only one variable is considered)
        self._inputs = [inputs] if isinstance(inputs, str) else inputs
        self._output = [output] if isinstance(output, str) else output

        Neuron.__init__(self, parameters=parameters, equations=equations, name=name, description=description)
        
        self._model_instantiated = False    # activated by BoldMonitor
