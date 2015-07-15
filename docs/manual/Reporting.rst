***********************************
Reporting
***********************************

ANNarchy includes an utility allowing to automatically generate a LaTeX report based onthe specifications provided in:

    Nordlie E, Gewaltig M-O, Plesser HE (2009). Towards Reproducible Descriptions of Neuronal Network Models. PLoS Comput Biol 5(8): e1000456.

The global method ``report()`` produces a ``.tex`` file (by default ``report.tex`` in the current directory, but this can be changed by passing the ``filename`` argument) which can be directly compiled with ``pdflatex`` or integrated into a larger file::

    report(filename="../model_description.tex")

Content of the file
===================

This report consists of different tables describing several aspects of the model:

1. **Summary**: A summary of the network, with a list of populations, neuron and synapse models, topologies, etc. This section may have to be adapted, as for example, ANNarchy does not make a distinction between synapse and plasticity models.

2. **Populations**: A list of populations, with their respective neural models and geometries.

3. **Projections**: A list of projections, with the pre- and post-synaptic populations, the target, the synapse model if any, and a description of the connection pattern.

4. **Neuron models**: For each neuron model, a description of its dynamics with equations parsed using SymPy and translated to the LaTeX mathematical language.

5. **Synapse models**: For each synapse model, a description of its dynamics if any.

6. **Population parameters**: The initial value (before the call to ``compile()``) of the parameters of each population. 

7. **Projection parameters**: The initial value of the parameters of each projections (if any).

8. **Input**: Inputs set to the network (has to be filled manually).

9. **Measurements**: Measurements done in the network (has to be filled manually). 


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
            tau = 10.0 : post-synaptic
        """,
        equations = """
            tau * dalpha/dt + alpha = pos(post.r - 1.0) : post-synaptic
            eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
        """, 
        name="Oja learning rule",
        description= """Rate-coded synapse implementing the Oja learning rule which ensures regularization of the synaptic weights."""
    ) 

3. Choose simple parameter and variable names for the description of equations. If a parameter/variable name uses only one character, it will be treated as a mathematical variable in the equations (ex: ``v`` becomes :math:`v`), otherwise the plain text representation will be used (ugly). If the name corresponds to a greek letter (``alpha``, ``tau``, etc.), it will be represented by the corresponding greek letter (:math:`\alpha`, :math:`\tau`). If the name is composed of two terms separated by an underscore (``tau_exc``), a subscript will be used (:math:`\tau_\text{exc}`). If more than one underscore is used, the text representation is used instead (LaTeX does not allow multiple subscripts).

