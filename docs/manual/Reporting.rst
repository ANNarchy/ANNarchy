***********************************
Reporting
***********************************

ANNarchy includes an utility allowing to automatically generate a report based on the current structure of the network::

    report(filename="model_description.tex")
    report(filename="model_description.md")

If the filename ends with ``.tex``, the LaTeX report will be generated based on the specifications provided in:

> Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

If the filename ends with ``.md``, the report will be generated in Markdown, so it can later be exported to pdf or html using `pandoc <http://www.pandoc.org>`_.

``report()`` accepts several arguments:

* ``filename``: name of the file where the report will be written (default: "./report.tex")
* ``standalone``: tells if the generated TeX file should be directly compilable or only includable. Ignored in Markdown.
* ``gather_subprojections``: if a projection between two populations has been implemented as a multiple of projections between sub-populations, this flag allows to group them in the summary (default: False).
* ``title``: title of the document (Markdown only)
* ``author``: author of the document (Markdown only)
* ``date``: date of the document (Markdown only)
* ``net_id``: id of the network to be used for reporting (default: 0, everything that was declared)

Content of the TeX file
========================

``report()`` produces a ``.tex`` file (by default ``report.tex`` in the current directory, but this can be changed by passing the ``filename`` argument) which can be directly compiled with ``pdflatex`` or integrated into a larger file::

    pdflatex model_description.tex

This report consists of different tables describing several aspects of the model:

1. **Summary**: A summary of the network, with a list of populations, neuron and synapse models, topologies, etc. This section may have to be adapted, as for example, ANNarchy does not make a distinction between synapse and plasticity models.

2. **Populations**: A list of populations, with their respective neural models and geometries.

3. **Projections**: A list of projections, with the pre- and post-synaptic populations, the target, the synapse model if any, and a description of the connection pattern.

4. **Neuron models**: For each neuron model, a description of its dynamics with equations parsed using SymPy and translated to the LaTeX mathematical language.

5. **Synapse models**: For each synapse model, a description of its dynamics if any.

6. **Parameters**: The initial value (before the call to ``compile()``) of the parameters of each population and projection (if any).

7. **Input**: Inputs set to the network (has to be filled manually).

8. **Measurements**: Measurements done in the network (has to be filled manually). 

Content of the Markdown file
===============================

The generated Mardown file is globally similar to the LaTeX one, with additional information that make it more useful for debugging (locality of attributes, type...). The Markown file is readable by design, but it can be translated to many markup languages (html, epub, latex, pdf...) using `pandoc <http://www.pandoc.org>`_.

To obtain a pdf from the Markdown file (supposing you have a LaTeX distribution available), just type:

.. code-block:: bash

    pandoc model_description.md -sN -V geometry:margin=1in -o model_description.pdf

The ``-V`` argument tells LaTex to use the full page instead of the default booklet format.

To obtain a html file, use:

.. code-block:: bash

    pandoc model_description.md -sSN --mathjax -o model_description.html

You can omit the ``-S`` option if you only want to include the code into a webpage, otherwise it is a standalone file. ``--mathjax`` is needed to display mathematical equations using the javascript library `MathJax <http://mathjax.org>`_.

By default, the html file has no styling, and tables can be very ugly. With a simple css file like `this one <../_static/simple.css>`_, the html page looks nicer (feel free to edit):

.. code-block:: bash

    pandoc model_description.md -sSN --mathjax --css=simple.css -o model_description.html

If you upload your model to a github-like service (bitbucket, gitlab, gogs...), it could be a good idea to generate the ``README.md`` directly with ``report()``. Do not forget to set a title+author+date then.


Documenting the network
========================

The report is generated based entirely on the Python script. For it to make sense, the user has to provide the necessary information while defining the network:

1. Populations must be assigned a unique name. If no name is given, generic names such as ``pop0`` or ``pop1`` will be used. If two populations have the same name, the connectivity will be unreadable::

    pop1 = Population(geometry=(100, 100), neuron=Izhikevich, name="Excitatory")
    pop2 = Population(geometry=(20, 20), neuron=Izhikevich, name="Inhibitory")

2. User-defined neuron and synapse models should be assigned a name and description. The name should be relatively short and generic (e.g. "Izhikevich", "BCM learning rule"), while the description should be more specific. They can contain LaTeX code, but remember to double the ``\`` which is the escape symbol in Python strings:

.. code-block:: python

    LIF = Neuron(
        parameters = """
            tau = 10.0
        """,
        equations = """
            tau * dv/dt + v = g_exc
        """,
        spike = "v > 30.0",
        reset = "v = 0.0"
        name = "LIF",
        description = "Leaky Integrate-and-Fire spiking neuron with time constant $\\tau$." 
    )

    Oja = Synapse(
        parameters = """
            eta = 10.0 
            tau = 10.0 : postsynaptic
        """,
        equations = """
            tau * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
            eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
        """, 
        name="Oja learning rule",
        description= """Oja learning rule ensuring regularization of the synaptic weights."""
    ) 

3. Choose simple parameter and variable names for the description of equations. If a parameter/variable name uses only one character, it will be treated as a mathematical variable in the equations (ex: ``v`` becomes :math:`v`), otherwise the plain text representation will be used (ugly). If the name corresponds to a greek letter (``alpha``, ``tau``, etc.), it will be represented by the corresponding greek letter (:math:`\alpha`, :math:`\tau`). If the name is composed of two terms separated by an underscore (``tau_exc``), a subscript will be used (:math:`\tau_\text{exc}`). If more than one underscore is used, the text representation is used instead (LaTeX does not allow multiple subscripts).

Example
===========

Let's take the homeostatic STDP ramp example provided in ``examples/homeostatic_stdp/Ramp.py`` and add names/descriptions to the objects:

.. code-block:: python

    from ANNarchy import *

    # Izhikevich RS neuron
    RSNeuron = Neuron(
        parameters = """
            a = 0.02 : population
            b = 0.2 : population
            c = -65. : population
            d = 8. : population
            tau_ampa = 5. : population
            tau_nmda = 150. : population
            vrev = 0.0 : population
        """ ,
        equations="""
            # Inputs
            I = g_ampa * (vrev - v) + g_nmda * nmda(v, -80.0, 60.0) * (vrev -v)  
            # Midpoint scheme      
            dv/dt = (0.04 * v + 5.0) * v + 140.0 - u + I : init=-65., midpoint
            du/dt = a * (b*v - u) : init=-13., midpoint
            # Izhikevich scheme
            # new_v = v + 0.5*(0.04 * v^2 + 5.0 * v + 140.0 - u + I) : init=-65.
            # v = new_v + 0.5*(0.04 * new_v^2 + 5.0 * new_v + 140.0 - u + I) : init=-65.
            # u += a * (b*v - u) : init=-13.
            # Conductances
            tau_ampa * dg_ampa/dt = -g_ampa : exponential
            tau_nmda * dg_nmda/dt = -g_nmda : exponential
        """ , 
        spike = """
            v >= 30.
        """, 
        reset = """
            v = c
            u += d
        """,
        functions = """
            nmda(v, t, s) = ((v-t)/(s))^2 / (1.0 + ((v-t)/(s))^2)
        """,
        name = "Regular-spiking Izhikevich",
        description = "Regular-spiking Izhikevich neuron, with AMPA/NMDA exponentially decreasing synapses."
    )

    # Input population
    inp = PoissonPopulation(100, rates=np.linspace(0.2, 20., 100), name="Poisson input")

    # RS neuron without homeostatic mechanism
    pop1 = Population(1, RSNeuron, name="RS neuron without homeostasis")
    pop1.compute_firing_rate(5000.)

    # RS neuron with homeostatic mechanism
    pop2 = Population(1, RSNeuron, name="RS neuron with homeostasis")
    pop2.compute_firing_rate(5000.)

    # Nearest Neighbour STDP
    nearest_neighbour_stdp = Synapse(
        parameters="""
            tau_plus = 20. : projection
            tau_minus = 60. : projection
            A_plus = 0.0002 : projection
            A_minus = 0.000066 : projection
            w_max = 0.03 : projection
        """,
        equations = """
            # Traces
            tau_plus  * dltp/dt = -ltp : exponential
            tau_minus * dltd/dt = -ltd : exponential
            # Nearest-neighbour
            w += if t_post >= t_pre: ltp else: - ltd : min=0.0, max=w_max
        """,
        pre_spike="""
            g_target += w
            ltp = A_plus
        """,         
        post_spike="""
            ltd = A_minus 
        """,
        name = "Nearest-neighbour STDP",
        description = "Nearest-neighbour STDP synaptic plasticity. Each synapse updates two traces (ltp and ltd) and updates continuously its weight."
    )

    # STDP with homeostatic regulation
    homeo_stdp = Synapse(
        parameters="""
            # STDP
            tau_plus = 20. : projection
            tau_minus = 60. : projection
            A_plus = 0.0002 : projection
            A_minus = 0.000066 : projection
            w_min = 0.0 : projection
            w_max = 0.03 : projection

            # Homeostatic regulation
            alpha = 0.1 : projection
            beta = 1.0 : projection
            gamma = 50. : projection
            Rtarget = 35. : projection
            T = 5000. : projection
        """,
        equations = """
            # Traces
            tau_plus  * dltp/dt = -ltp : exponential
            tau_minus * dltd/dt = -ltd : exponential
            # Homeostatic values
            R = post.r : postsynaptic
            K = R/(T*(1.+fabs(1. - R/Rtarget) * gamma)) : postsynaptic
            # Nearest-neighbour
            stdp = if t_post >= t_pre: ltp else: - ltd 
            w += (alpha * w * (1- R/Rtarget) + beta * stdp ) * K : min=w_min, max=w_max
        """,
        pre_spike="""
            g_target += w
            ltp = A_plus
        """,         
        post_spike="""
            ltd = A_minus 
        """ ,
        name = "Nearest-neighbour STDP with homeostasis",
        description = "Nearest-neighbour STDP synaptic plasticity with an additional homeostatic term. "
    )

    # Projection without homeostatic mechanism
    proj1 = Projection(inp, pop1, ['ampa', 'nmda'], synapse=nearest_neighbour_stdp)
    proj1.connect_all_to_all(Uniform(0.01, 0.03))

    # Projection with homeostatic mechanism
    proj2 = Projection(inp, pop2, ['ampa', 'nmda'], synapse=homeo_stdp)
    proj2.connect_all_to_all(weights=Uniform(0.01, 0.03))


    # Record
    m1 = Monitor(pop1, 'r')
    m2 = Monitor(pop2, 'r')

    report('ramp.md', 
            title="Biologically plausible models of homeostasis and STDP; Stability and learning in spiking neural networks", 
            author="Carlson, Richert, Dutt and Krichmar",
            date="Neural Networks (IJCNN) 2013")

This generates the following Markdown file:

.. code-block:: md

    ---
    title: Biologically plausible models of homeostasis and STDP; Stability and learning in spiking neural networks
    author: Carlson, Richert, Dutt and Krichmar
    date: Neural Networks (IJCNN) 2013
    ---

    # Structure of the network

    * ANNarchy 4.6.2b using the default backend.
    * Numerical step size: 1.0 ms.

    ## Populations

    | **Population**                | **Size** | **Neuron type**            | 
    | ----------------------------- | -------- | -------------------------- | 
    | Poisson input                 | 100      | Poisson                    | 
    | RS neuron without homeostasis | 1        | Regular-spiking Izhikevich | 
    | RS neuron with homeostasis    | 1        | Regular-spiking Izhikevich | 


    ## Projections

    | **Source**    | **Destination**               | **Target**  | **Synapse type**                        | **Pattern**                                               | 
    | ------------- | ----------------------------- | ----------- | --------------------------------------- | --------------------------------------------------------- | 
    | Poisson input | RS neuron without homeostasis | ampa / nmda | Nearest-neighbour STDP                  | All-to-All, weights $\mathcal{U}$(0.01, 0.03), delays 0.0 | 
    | Poisson input | RS neuron with homeostasis    | ampa / nmda | Nearest-neighbour STDP with homeostasis | All-to-All, weights $\mathcal{U}$(0.01, 0.03), delays 0.0 | 


    ## Monitors

    | **Object**                    | **Variables** | **Period** | 
    | ----------------------------- | ------------- | ---------- | 
    | RS neuron without homeostasis | r             | 1.0        | 
    | RS neuron with homeostasis    | r             | 1.0        | 


    # Neuron models

    ## Regular-spiking Izhikevich

    Regular-spiking Izhikevich neuron, with AMPA/NMDA exponentially decreasing synapses.

    **Parameters:**

    | **Name**             | **Default value** | **Locality**   | **Type** | 
    | -------------------- | ----------------- | -------------- | -------- | 
    | $a$                  | 0.02              | per population | double   | 
    | $b$                  | 0.2               | per population | double   | 
    | $c$                  | -65.0             | per population | double   | 
    | $d$                  | 8.0               | per population | double   | 
    | $\tau_{\text{ampa}}$ | 5.0               | per population | double   | 
    | $\tau_{\text{nmda}}$ | 150.0             | per population | double   | 
    | ${\text{vrev}}$      | 0.0               | per population | double   | 

    **Equations:**

    * Variable $I$ : per neuron, initial value: 0.0

    $$
    {I}(t) = {g_{\text{ampa}}}(t) \cdot \left({\text{vrev}} - {v}(t)\right) + {g_{\text{nmda}}}(t) \cdot \left({\text{vrev}} - {v}(t)\right) \cdot \operatorname{nmda}{\left ({v}(t),-80.0,60.0 \right )}
    $$

    * Variable $v$ : per neuron, initial value: -65.0, midpoint numerical method

    $$
    \frac{d{v}(t)}{dt} = {I}(t) - {u}(t) + {v}(t) \cdot \left(0.04 \cdot {v}(t) + 5.0\right) + 140.0
    $$

    * Variable $u$ : per neuron, initial value: -13.0, midpoint numerical method

    $$
    \frac{d{u}(t)}{dt} = a \cdot \left(b \cdot {v}(t) - {u}(t)\right)
    $$

    * Variable $g_{\text{ampa}}$ : per neuron, initial value: 0.0, exponential numerical method

    $$
    \frac{d{g_{\text{ampa}}}(t)}{dt} \cdot \tau_{\text{ampa}} = - {g_{\text{ampa}}}(t)
    $$

    * Variable $g_{\text{nmda}}$ : per neuron, initial value: 0.0, exponential numerical method

    $$
    \frac{d{g_{\text{nmda}}}(t)}{dt} \cdot \tau_{\text{nmda}} = - {g_{\text{nmda}}}(t)
    $$

    **Spike emission:**

    if ${v}(t) \geq 30.0$ :

    * Emit a spike a time $t$.
    * ${v}(t) = c$
    * ${u}(t) \mathrel{+}= d$


    **Functions**

    $${\text{nmda}}(v, t, s) = \frac{\left(- t + v\right)^{2}}{s^{2} \cdot \left(1.0 + \frac{1}{s^{2}} \cdot \left(- t + v\right)^{2}\right)}$$


    ## Poisson

    Spiking neuron with spikes emitted according to a Poisson distribution.

    **Parameters:**

    | **Name**         | **Default value** | **Locality** | **Type** | 
    | ---------------- | ----------------- | ------------ | -------- | 
    | ${\text{rates}}$ | 10.0              | per neuron   | double   | 

    **Equations:**

    * Variable $p$ : per neuron, initial value: 0.0

    $$
    {p}(t) = \frac{1000.0}{\Delta t} \cdot \mathcal{U}{\left (0.0,1.0 \right )}
    $$

    **Spike emission:**

    if ${p}(t) < {\text{rates}}$ :

    * Emit a spike a time $t$.

    # Synapse models

    ## Nearest-neighbour STDP

    Nearest-neighbour STDP synaptic plasticity. Each synapse updates two traces (ltp and ltd) and updates continuously its weight.

    **Parameters:**

    | **Name**              | **Default value** | **Locality**   | **Type** | 
    | --------------------- | ----------------- | -------------- | -------- | 
    | $\tau_{\text{plus}}$  | 20.0              | per projection | double   | 
    | $\tau_{\text{minus}}$ | 60.0              | per projection | double   | 
    | $A_{\text{plus}}$     | 0.0002            | per projection | double   | 
    | $A_{\text{minus}}$    | 6.6e-05           | per projection | double   | 
    | $w_{\text{max}}$      | 0.03              | per projection | double   | 

    **Equations:**

    * Variable ${\text{ltp}}$ : per synapse, initial value: 0.0, exponential numerical method

    $$
    \frac{d{{\text{ltp}}}(t)}{dt} \cdot \tau_{\text{plus}} = - {{\text{ltp}}}(t)
    $$

    * Variable ${\text{ltd}}$ : per synapse, initial value: 0.0, exponential numerical method

    $$
    \frac{d{{\text{ltd}}}(t)}{dt} \cdot \tau_{\text{minus}} = - {{\text{ltd}}}(t)
    $$

    * Variable $w$ : per synapse, initial value: 0.0, minimum: 0.0, maximum: w_max

    $$
    {w}(t) \mathrel{+}= \begin{cases}{{\text{ltp}}}(t)\qquad \text{if} \quad t_{\text{pos}} \geq t_{\text{pre}}\\ - {{\text{ltd}}}(t) \qquad \text{otherwise.} \end{cases}
    $$

    **Pre-synaptic event at $t_\text{pre} + d$:**
    $$g_{\text{target}(t)} \mathrel{+}= {w}(t)$$
    $${{\text{ltp}}}(t) = A_{\text{plus}}$$

    **Post-synaptic event at $t_\text{post}$:**
    $${{\text{ltd}}}(t) = A_{\text{minus}}$$

    ## Nearest-neighbour STDP with homeostasis

    Nearest-neighbour STDP synaptic plasticity with an additional homeostatic term. 

    **Parameters:**

    | **Name**              | **Default value** | **Locality**   | **Type** | 
    | --------------------- | ----------------- | -------------- | -------- | 
    | $\tau_{\text{plus}}$  | 20.0              | per projection | double   | 
    | $\tau_{\text{minus}}$ | 60.0              | per projection | double   | 
    | $A_{\text{plus}}$     | 0.0002            | per projection | double   | 
    | $A_{\text{minus}}$    | 6.6e-05           | per projection | double   | 
    | $w_{\text{min}}$      | 0.0               | per projection | double   | 
    | $w_{\text{max}}$      | 0.03              | per projection | double   | 
    | $\alpha$              | 0.1               | per projection | double   | 
    | $\beta$               | 1.0               | per projection | double   | 
    | $\gamma$              | 50.0              | per projection | double   | 
    | ${\text{Rtarget}}$    | 35.0              | per projection | double   | 
    | $T$                   | 5000.0            | per projection | double   | 

    **Equations:**

    * Variable ${\text{ltp}}$ : per synapse, initial value: 0.0, exponential numerical method

    $$
    \frac{d{{\text{ltp}}}(t)}{dt} \cdot \tau_{\text{plus}} = - {{\text{ltp}}}(t)
    $$

    * Variable ${\text{ltd}}$ : per synapse, initial value: 0.0, exponential numerical method

    $$
    \frac{d{{\text{ltd}}}(t)}{dt} \cdot \tau_{\text{minus}} = - {{\text{ltd}}}(t)
    $$

    * Variable $R$ : per post-synaptic neuron, initial value: 0.0

    $$
    {R}(t) = {{r}^{\text{post}}}(t)
    $$

    * Variable $K$ : per post-synaptic neuron, initial value: 0.0

    $$
    {K}(t) = \frac{{R}(t)}{T \cdot \left(\gamma \cdot \left|{f}\right|{\left (- \frac{{R}(t)}{{\text{Rtarget}}} + 1.0 \right )} + 1.0\right)}
    $$

    * Variable ${\text{stdp}}$ : per synapse, initial value: 0.0

    $$
    {{\text{stdp}}}(t) = \begin{cases}{{\text{ltp}}}(t)\qquad \text{if} \quad t_{\text{pos}} \geq t_{\text{pre}}\\ - {{\text{ltd}}}(t) \qquad \text{otherwise.} \end{cases}
    $$

    * Variable $w$ : per synapse, initial value: 0.0, minimum: w_min, maximum: w_max

    $$
    {w}(t) \mathrel{+}= {K}(t) \cdot \left(\alpha \cdot {w}(t) \cdot \left(- \frac{{R}(t)}{{\text{Rtarget}}} + 1\right) + \beta \cdot {{\text{stdp}}}(t)\right)
    $$

    **Pre-synaptic event at $t_\text{pre} + d$:**
    $$g_{\text{target}(t)} \mathrel{+}= {w}(t)$$
    $${{\text{ltp}}}(t) = A_{\text{plus}}$$

    **Post-synaptic event at $t_\text{post}$:**
    $${{\text{ltd}}}(t) = A_{\text{minus}}$$

    # Parameters

    ## Population parameters

    | **Population**                | **Neuron type**            | **Name**             | **Value**     | 
    | ----------------------------- | -------------------------- | -------------------- | ------------- | 
    | Poisson input                 | Poisson                    | ${\text{rates}}$     | $[0.2, 20.0]$ | 
    | RS neuron without homeostasis | Regular-spiking Izhikevich | $a$                  | 0.02          | 
    |                               |                            | $b$                  | 0.2           | 
    |                               |                            | $c$                  | -65.0         | 
    |                               |                            | $d$                  | 8.0           | 
    |                               |                            | $\tau_{\text{ampa}}$ | 5.0           | 
    |                               |                            | $\tau_{\text{nmda}}$ | 150.0         | 
    |                               |                            | ${\text{vrev}}$      | 0.0           | 
    | RS neuron with homeostasis    | Regular-spiking Izhikevich | $a$                  | 0.02          | 
    |                               |                            | $b$                  | 0.2           | 
    |                               |                            | $c$                  | -65.0         | 
    |                               |                            | $d$                  | 8.0           | 
    |                               |                            | $\tau_{\text{ampa}}$ | 5.0           | 
    |                               |                            | $\tau_{\text{nmda}}$ | 150.0         | 
    |                               |                            | ${\text{vrev}}$      | 0.0           | 


    ## Projection parameters

    | **Projection**                                                                     | **Synapse type**                        | **Name**              | **Value** | 
    | ---------------------------------------------------------------------------------- | --------------------------------------- | --------------------- | --------- | 
    | Poisson input  $\rightarrow$ RS neuron without homeostasis with target ampa / nmda | Nearest-neighbour STDP                  | $\tau_{\text{plus}}$  | 20.0      | 
    |                                                                                    |                                         | $\tau_{\text{minus}}$ | 60.0      | 
    |                                                                                    |                                         | $A_{\text{plus}}$     | 0.0002    | 
    |                                                                                    |                                         | $A_{\text{minus}}$    | 6.6e-05   | 
    |                                                                                    |                                         | $w_{\text{max}}$      | 0.03      | 
    | Poisson input  $\rightarrow$ RS neuron with homeostasis with target ampa / nmda    | Nearest-neighbour STDP with homeostasis | $\tau_{\text{plus}}$  | 20.0      | 
    |                                                                                    |                                         | $\tau_{\text{minus}}$ | 60.0      | 
    |                                                                                    |                                         | $A_{\text{plus}}$     | 0.0002    | 
    |                                                                                    |                                         | $A_{\text{minus}}$    | 6.6e-05   | 
    |                                                                                    |                                         | $w_{\text{min}}$      | 0.0       | 
    |                                                                                    |                                         | $w_{\text{max}}$      | 0.03      | 
    |                                                                                    |                                         | $\alpha$              | 0.1       | 
    |                                                                                    |                                         | $\beta$               | 1.0       | 
    |                                                                                    |                                         | $\gamma$              | 50.0      | 
    |                                                                                    |                                         | ${\text{Rtarget}}$    | 35.0      | 
    |                                                                                    |                                         | $T$                   | 5000.0    | 


Once transformed by pandoc, this generates for example the following html code:

:Title: Biologically plausible models of homeostasis and STDP; Stability and learning in spiking neural networks
:Author: Carlson, Richert, Dutt and Krichmar
:Date:   Neural Networks (IJCNN) 2013


**1. Structure of the network**

-  ANNarchy 4.6.3 using the default backend.
-  Numerical step size: 1.0 ms.

**1.1 Populations**

+---------------------------------+------------+------------------------------+
| **Population**                  | **Size**   | **Neuron type**              |
+=================================+============+==============================+
| Poisson input                   | 100        | Poisson                      |
+---------------------------------+------------+------------------------------+
| RS neuron without homeostasis   | 1          | Regular-spiking Izhikevich   |
+---------------------------------+------------+------------------------------+
| RS neuron with homeostasis      | 1          | Regular-spiking Izhikevich   |
+---------------------------------+------------+------------------------------+

**1.2 Projections**

+-------------+------------------+------------+--------------------+-----------------------------+
| **Source**  | **Destination**  | **Target** | **Synapse type**   | **Pattern**                 |
|             |                  |            |                    |                             |
+=============+==================+============+====================+=============================+
| Poisson     | RS neuron        | ampa       | Nearest-neighbour  | All-to-All, weights         |
| input       | without          | /          | STDP               | :math:`\mathcal{U}`\ (0.01, |
|             | homeostasis      | nmda       |                    | 0.03), delays 0.0           |
+-------------+------------------+------------+--------------------+-----------------------------+
| Poisso      | RS neuron with   | ampa       | Nearest-neighbour  | All-to-All, weights         |
| n           | homeostasis      | /          | STDP with          | :math:`\mathcal{U}`\ (0.01, |
| input       |                  | nmda       | homeostasis        | 0.03), delays 0.0           |
+-------------+------------------+------------+--------------------+-----------------------------+

**1.3 Monitors**

+---------------------------------+-----------------+--------------+
| **Object**                      | **Variables**   | **Period**   |
+=================================+=================+==============+
| RS neuron without homeostasis   | r               | 1.0          |
+---------------------------------+-----------------+--------------+
| RS neuron with homeostasis      | r               | 1.0          |
+---------------------------------+-----------------+--------------+

**2 Neuron models**

**2.1 Regular-spiking Izhikevich**

Regular-spiking Izhikevich neuron, with AMPA/NMDA exponentially
decreasing synapses.

**Parameters:**

+------------------------------+---------------------+------------------+------------+
| **Name**                     | **Default value**   | **Locality**     | **Type**   |
+==============================+=====================+==================+============+
| :math:`a`                    | 0.02                | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`b`                    | 0.2                 | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`c`                    | -65.0               | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`d`                    | 8.0                 | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`\tau_{\text{ampa}}`   | 5.0                 | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`\tau_{\text{nmda}}`   | 150.0               | per population   | double     |
+------------------------------+---------------------+------------------+------------+
| :math:`{\text{vrev}}`        | 0.0                 | per population   | double     |
+------------------------------+---------------------+------------------+------------+

**Equations:**

-  Variable :math:`I` : per neuron, initial value: 0.0

.. math::


   {I}(t) = {g_{\text{ampa}}}(t) \cdot \left({\text{vrev}} - {v}(t)\right) + {g_{\text{nmda}}}(t) \cdot \left({\text{vrev}} - {v}(t)\right) \cdot \operatorname{nmda}{\left ({v}(t),-80.0,60.0 \right )}

-  Variable :math:`v` : per neuron, initial value: -65.0, midpoint
   numerical method

.. math::


   \frac{d{v}(t)}{dt} = {I}(t) - {u}(t) + {v}(t) \cdot \left(0.04 \cdot {v}(t) + 5.0\right) + 140.0

-  Variable :math:`u` : per neuron, initial value: -13.0, midpoint
   numerical method

.. math::


   \frac{d{u}(t)}{dt} = a \cdot \left(b \cdot {v}(t) - {u}(t)\right)

-  Variable :math:`g_{\text{ampa}}` : per neuron, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{g_{\text{ampa}}}(t)}{dt} \cdot \tau_{\text{ampa}} = - {g_{\text{ampa}}}(t)

-  Variable :math:`g_{\text{nmda}}` : per neuron, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{g_{\text{nmda}}}(t)}{dt} \cdot \tau_{\text{nmda}} = - {g_{\text{nmda}}}(t)

**Spike emission:**

if :math:`{v}(t) \geq 30.0` :

-  Emit a spike a time :math:`t`.
-  :math:`{v}(t) = c`
-  :math:`{u}(t) \mathrel{+}= d`

**Functions**

.. math:: {\text{nmda}}(v, t, s) = \frac{\left(- t + v\right)^{2}}{s^{2} \cdot \left(1.0 + \frac{1}{s^{2}} \cdot \left(- t + v\right)^{2}\right)}

**2.2 Poisson**

Spiking neuron with spikes emitted according to a Poisson distribution.

**Parameters:**

+--------------------------+---------------------+----------------+------------+
| **Name**                 | **Default value**   | **Locality**   | **Type**   |
+==========================+=====================+================+============+
| :math:`{\text{rates}}`   | 10.0                | per neuron     | double     |
+--------------------------+---------------------+----------------+------------+

**Equations:**

-  Variable :math:`p` : per neuron, initial value: 0.0

.. math::


   {p}(t) = \frac{1000.0}{\Delta t} \cdot \mathcal{U}{\left (0.0,1.0 \right )}

**Spike emission:**

if :math:`{p}(t) < {\text{rates}}` :

-  Emit a spike a time :math:`t`.

**3 Synapse models**

**3.1 Nearest-neighbour STDP**

Nearest-neighbour STDP synaptic plasticity. Each synapse updates two
traces (ltp and ltd) and updates continuously its weight.

**Parameters:**

+-------------------------------+---------------------+------------------+------------+
| **Name**                      | **Default value**   | **Locality**     | **Type**   |
+===============================+=====================+==================+============+
| :math:`\tau_{\text{plus}}`    | 20.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`\tau_{\text{minus}}`   | 60.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`A_{\text{plus}}`       | 0.0002              | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`A_{\text{minus}}`      | 6.6e-05             | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`w_{\text{max}}`        | 0.03                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+

**Equations:**

-  Variable :math:`{\text{ltp}}` : per synapse, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{{\text{ltp}}}(t)}{dt} \cdot \tau_{\text{plus}} = - {{\text{ltp}}}(t)

-  Variable :math:`{\text{ltd}}` : per synapse, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{{\text{ltd}}}(t)}{dt} \cdot \tau_{\text{minus}} = - {{\text{ltd}}}(t)

-  Variable :math:`w` : per synapse, initial value: 0.0, minimum: 0.0,
   maximum: w\_max

.. math::


   {w}(t) \mathrel{+}= \begin{cases}{{\text{ltp}}}(t)\qquad \text{if} \quad t_{\text{pos}} \geq t_{\text{pre}}\\ - {{\text{ltd}}}(t) \qquad \text{otherwise.} \end{cases}

**Pre-synaptic event at :math:`t_\text{pre} + d`:**

.. math:: g_{\text{target}(t)} \mathrel{+}= {w}(t)

.. math:: {{\text{ltp}}}(t) = A_{\text{plus}}

**Post-synaptic event at :math:`t_\text{post}`:**

.. math:: {{\text{ltd}}}(t) = A_{\text{minus}}

**3.2 Nearest-neighbour STDP with homeostasis**

Nearest-neighbour STDP synaptic plasticity with an additional
homeostatic term.

**Parameters:**

+-------------------------------+---------------------+------------------+------------+
| **Name**                      | **Default value**   | **Locality**     | **Type**   |
+===============================+=====================+==================+============+
| :math:`\tau_{\text{plus}}`    | 20.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`\tau_{\text{minus}}`   | 60.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`A_{\text{plus}}`       | 0.0002              | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`A_{\text{minus}}`      | 6.6e-05             | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`w_{\text{min}}`        | 0.0                 | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`w_{\text{max}}`        | 0.03                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`\alpha`                | 0.1                 | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`\beta`                 | 1.0                 | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`\gamma`                | 50.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`{\text{Rtarget}}`      | 35.0                | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+
| :math:`T`                     | 5000.0              | per projection   | double     |
+-------------------------------+---------------------+------------------+------------+

**Equations:**

-  Variable :math:`{\text{ltp}}` : per synapse, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{{\text{ltp}}}(t)}{dt} \cdot \tau_{\text{plus}} = - {{\text{ltp}}}(t)

-  Variable :math:`{\text{ltd}}` : per synapse, initial value: 0.0,
   exponential numerical method

.. math::


   \frac{d{{\text{ltd}}}(t)}{dt} \cdot \tau_{\text{minus}} = - {{\text{ltd}}}(t)

-  Variable :math:`R` : per post-synaptic neuron, initial value: 0.0

.. math::


   {R}(t) = {{r}^{\text{post}}}(t)

-  Variable :math:`K` : per post-synaptic neuron, initial value: 0.0

.. math::


   {K}(t) = \frac{{R}(t)}{T \cdot \left(\gamma \cdot \left|{f}\right|{\left (- \frac{{R}(t)}{{\text{Rtarget}}} + 1.0 \right )} + 1.0\right)}

-  Variable :math:`{\text{stdp}}` : per synapse, initial value: 0.0

.. math::


   {{\text{stdp}}}(t) = \begin{cases}{{\text{ltp}}}(t)\qquad \text{if} \quad t_{\text{pos}} \geq t_{\text{pre}}\\ - {{\text{ltd}}}(t) \qquad \text{otherwise.} \end{cases}

-  Variable :math:`w` : per synapse, initial value: 0.0, minimum:
   w\_min, maximum: w\_max

.. math::


   {w}(t) \mathrel{+}= {K}(t) \cdot \left(\alpha \cdot {w}(t) \cdot \left(- \frac{{R}(t)}{{\text{Rtarget}}} + 1\right) + \beta \cdot {{\text{stdp}}}(t)\right)

**Pre-synaptic event at :math:`t_\text{pre} + d`:**

.. math:: g_{\text{target}(t)} \mathrel{+}= {w}(t)

.. math:: {{\text{ltp}}}(t) = A_{\text{plus}}

**Post-synaptic event at :math:`t_\text{post}`:**

.. math:: {{\text{ltd}}}(t) = A_{\text{minus}}

**4 Parameters**

**4.1 Population parameters**

+---------------------------------+------------------------------+------------------------------+-----------------------+
| **Population**                  | **Neuron type**              | **Name**                     | **Value**             |
+=================================+==============================+==============================+=======================+
| Poisson input                   | Poisson                      | :math:`{\text{rates}}`       | :math:`[0.2, 20.0]`   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
| RS neuron without homeostasis   | Regular-spiking Izhikevich   | :math:`a`                    | 0.02                  |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`b`                    | 0.2                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`c`                    | -65.0                 |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`d`                    | 8.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`\tau_{\text{ampa}}`   | 5.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`\tau_{\text{nmda}}`   | 150.0                 |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`{\text{vrev}}`        | 0.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
| RS neuron with homeostasis      | Regular-spiking Izhikevich   | :math:`a`                    | 0.02                  |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`b`                    | 0.2                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`c`                    | -65.0                 |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`d`                    | 8.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`\tau_{\text{ampa}}`   | 5.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`\tau_{\text{nmda}}`   | 150.0                 |
+---------------------------------+------------------------------+------------------------------+-----------------------+
|                                 |                              | :math:`{\text{vrev}}`        | 0.0                   |
+---------------------------------+------------------------------+------------------------------+-----------------------+

**4.2 Projection parameters**

+----------------------------------------+--------------------+--------------------------------------------+-----------------+
| **Projection**                         | **Synapse type**   | **Name**                                   | **Value**       |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+========================================+====================+============================================+=================+
| Poisson input :math:`\rightarrow` RS   | Nearest-neighbour  | :math:`\tau_{\text{plus}}`                 | 20.0            |
| neuron without homeostasis with target | STDP               |                                            |                 |
| ampa / nmda                            |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`\tau_{\text{minus}}`                | 60.0            |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`A_{\text{plus}}`                    | 0.0002          |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`A_{ \text{minus}}`                  | 6.6e-5          |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`w_{ \text{max}}`                    | 0.03            |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
| Poisson input :math:`\rightarrow` RS   | Nearest-neighbour  | :math:`\tau_{\text{plus}}`                 | 20.0            |
| neuron with homeostasis with target    | STDP with          |                                            |                 |
| ampa / nmda                            | homeostasis        |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`\tau_{\text{minus}}`                | 60.0            |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`A_{\text{plus}}`                    | 0.0002          |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`A_{\text{minus}}`                   | 6.6e-5          |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`w_{\text{min}}`                     | 0.0             |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`w_{\text{max}}`                     | 0.03            |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`\alpha`                             | 0.1             |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`\beta`                              | 1.0             |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`\gamma`                             | 50.0            |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`{\text{Rtarget}}`                   | 35.0            |
|                                        |                    |                                            |                 |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
|                                        |                    | :math:`T`                                  | 5000            |
|                                        |                    |                                            |                 |
+----------------------------------------+--------------------+--------------------------------------------+-----------------+
