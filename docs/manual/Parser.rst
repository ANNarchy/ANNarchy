**************************
Parser
**************************

A Neuron or Synapse type is primarily defined by two sets of values which must be specified in its constructor:

* **Parameters** are values such as time constants which are constant during the simulation. They can be the same throughout the population/projection, or take different values.

* **Variables** are neuronal variables (for example the membrane potential or firing rate) or synaptic variables (the synaptic efficiency) whose value evolve with time during the simulation. The equation (whether it is an ordinary differential equation or not) ruling their evolution can be described using a specific meta-language.


Parameters
---------------------

Parameters are defined by a multi-string consisting of one or more parameter definitions:

.. code-block:: python

    parameters = """
        tau = 10.0
        eta = 0.5
    """

Each parameter should be defined on a single line, with its name on the left side of the equal sign, and its value on the right side. The given value corresponds to the  initial value of the parameter (but it can be changed at any further point of the simulation).

As a neuron/synapse type is likely to be reused in different populations/projections, it is good practice to set reasonable initial values in the neuron/synapse type, and eventually adapt them to the corresponding populations/projections later on.


**Local vs. global parameters**

By default, a neural parameter will be unique to each neuron (i.e. each neuron instance will hold a copy of the parameter) or synapse. In order to save memory space, one can force ANNarchy to store only one parameter value for a whole population by specifying the ``population`` flag after a ``:`` symbol following the parameter definition:

.. code-block:: python

    parameters = """
        tau = 10.0
        eta = 0.5 : population
    """

In this case, there will be only only one instance of the ``eta`` parameter for the whole population. ``eta`` is called a **global** parameter, in opposition to **local** parameters which are the default.

The same is true for synapses, whose parameters are unique to each synapse in a given projection. If the ``post-synaptic`` flag is passed, the parameter will be common to all synapses of a post-synaptic neuron.

**Type of the variable**

Parameters have floating-point precision by default. If you want to force the parameter to be an integer or boolean, you can also pass the ``int`` and ``bool`` flags, separated by commas:

.. code-block:: python

    parameters = """
        tau = 10.0
        eta = 1 : population, int
    """

Variables
--------------------

Time-varying variables are also defined using a multi-line description:

.. code-block:: python

    equations = """
        noise = Uniform(0.0, 0.2)
        tau * dmp/dt  + mp = baseline + sum(exc) + noise
        r = pos(mp)
    """

The evolution of each variable with time can be described through a simple equation or an ordinary differential equation (ODE). ANNarchy provides a simple parser for mathematical expressions, whose role is to translate a high-level description of the equation into an optimized C++ code snippet.

The equation for one variable can depend on parameters, other variables (even when declared later) or numerical constants. Variables are updated in the same order as their declaration in the multistring.

The declaration of a single variable can extend on multiple lines:

.. code-block:: python

    equations = """
        noise = Uniform(0.0, 0.2)
        tau * dmp/dt  = baseline - mp
                        + sum(exc) + noise : max = 1.0
        rate = pos(mp)
    """

As it is only a parser and not a solver, some limitations exist:

* Simple equations must hold only the name of the variable on the left sign of the equation. Variable definitions such as ``rate + mp = noise`` are forbidden, as it would be impossible to guess which variable should be updated.

* ODEs are more free regarding the left side, but only one variable should hold the gradient: the one which will be updated. The following definitions are equivalent and will lead to the same C++ code:


.. code-block:: python

    tau * dmp/dt  = baseline - mp

    tau * dmp/dt  + mp = baseline

    tau * dmp/dt  + mp -  baseline = 0

    dmp/dt  = (baseline - mp) / tau

In practice, ODEs are transformed using Sympy into the last form (only the gradient stays on the left) and numerized using the chosen numerical method.


Constraints
____________

**Locality and type**

Like the parameters, variables also accept the ``population`` and ``post-synaptic`` to define the local/global character of the variable, as well as the ``int`` or ``bool`` flags for their type.

**Initial value**

The initial value of the variable (before the first simulation starts) can also be specified using the ``init`` keyword followed by the desired value:


.. code-block:: python

    equations = """
        tau * dmp/dt + mp = baseline : init = 0.2
    """

It must be a single value (the same for all neurons in the population or all synapses in the projeciton) and should not depend on other parameters and variables. This initial value can be specifically changed after the ``Population`` or ``Projection`` objects are created (see :doc:`Populations`).

**Min and Max values of a variable**

Upper- and lower-bounds can be set using the ``min`` and ``max`` keywords:

.. code-block:: python

    equations = """
        tau * dmp/dt  + mp = baseline : min = -0.2, max = 1.0
    """

At each step of the simulation, after the update rule is calculated for ``mp``, the new value will be compared to the ``min`` and ``max`` value, and clamped if necessary.

``min`` and ``max`` can be single values, parameters, variables or functions of all these:

.. code-block:: python

    parameters = """
        tau = 10.0
        min_mp = -1.0 : population
        max_mp = 1.0
    """,
    equations = """
        variance = Uniform(0.0, 1.0)
        tau * dmp/dt  + mp = sum(exc) : min = min_mp, max = max_mp + variance
        r = mp : min = 0.0 # Equivalent to r = pos(mp)
    """


**Numerical method**

The numerization method for a single ODEs can be explicitely set by specifying a flag::

    tau * dmp/dt  + mp = sum(exc) : exponential

The available numerical methods are described in :doc:`NumericalMethods`.



**Summary of allowed keywords for variables:**

* *init*: defines the initialization value at begin of simulation and after a network reset (default: 0.0)
* *min*: minimum allowed value  (unset by default)
* *max*: maximum allowed value (unset by default)
* *population*: the attribute is equal for all neurons in a population.
* *post-synaptic*: the attribute is equal for all synapses of a post-synaptic neuron.
* *explicit*, *implicit*, *exponential*, *midpoint*, *event-driven*: the numerical method to be used.

Allowed vocabulary
-------------------

The mathematical parser relies heavily on the one provided by `SymPy <http://sympy.org/>`_.

Constants
_________

All parameters and variables use implicitly the floating-point double precision, except when stated otherwise with the ``int`` or ``bool`` keywords. You can use numerical constants within the equation, noting that they will be automatically converted to this precision:

.. code-block:: python

    tau * dmp / dt  = 1 / pos(mp) + 1

The constant :math:`\pi` is available under the literal form ``pi``.

Operators
__________

* Additions (+), substractions (-), multiplications (*), divisions (/) and power functions (^) are of course allowed.

* Gradients are allowed only for the variable currently described. They take the form:

.. code-block:: python

    dmp / dt  = A

with a ``d`` preceding the variable's name and terminated by ``/dt`` (with or without spaces). Gradients must be on the left side of the equation.

* To update the value of a variable at each time step, the operators ``=``, ``+=``, ``-=``, ``*=``, and ``/=`` are allowed.


Parameters and Variables
_________________________

Any parameter or variable defined in the same Neuron/Synapse can be used inside another equation. Additionally, the following variables are pre-defined:

* ``dt`` : the discretization time step for the simulation. Using this variable, you can define the numerical method by yourself. For example:

.. code-block:: python

    tau * dmp / dt  + mp = baseline

with backward Euler would be equivalent to:

.. code-block:: python

    mp += dt/tau * (baseline -mp)

* ``t`` : the time in milliseconds elapsed since the creation of the network. This allows to generate oscillating variables:

.. code-block:: python

    f = 10.0 # Frequency of 10 Hz
    phi = pi/4 # Phase
    ts = t / 1000.0 # ts is in seconds
    r = 10.0 * (sin(2*pi*f*ts + phi) + 1.0)


Random number generators
_________________________

Several random generators are available and can be used within an equation. In the current version are for example available:

* ``Uniform(min, max)`` generates random numbers from a uniform distribution in the range :math:`[\text{min}, \text{max}]`.

* ``Normal(mu, sigma)`` generates random numbers from a normal distribution with min mu and standard deviation sigma.

See :doc:`../API/RandomDistribution` for more distributions. For example:

.. code-block:: python

    noise = Uniform(-0.5, 0.5)

The arguments to the random distributions can be either fixed values or (functions of) global parameters.

.. code-block:: python

    min_val = -0.5 : population
    max_val = 0.5 : population
    noise = Uniform(min_val, max_val)


It is not allowed to use local parameters (with different values per neuron) or variables, as the random number generators are initialized only once at network creation (doing otherwise would impair performance too much). If a global parameter is used, changing its value will not affect the generator after compilation.

It is therefore better practice to use normalized random generators and scale their outputs:

.. code-block:: python

    min_val = -0.5 : population
    max_val = 0.5 : population
    noise = min_val + (max_val - min_val) * Uniform(0.0, 1.0)


Mathematical functions
_______________________

* Most mathematical functions of the ``cmath`` library are understood by the parser, for example:

.. code-block:: python

    cos, sin, tan, acos, asin, atan, exp, abs, fabs, sqrt, log, ln

* The positive and negative parts of a term are also defined, with short and long versions:

.. code-block:: python

    r = pos(mp)
    r = positive(mp)
    r = neg(mp)
    r = negative(mp)

* A piecewise linear function is also provided (linear when x is between a and b, saturated at a or b otherwise):

.. code-block:: python

    r = clip(x, a, b)

* For integer variables, the modulo operator is defined:

.. code-block::python

    x += 1 : int
    y = modulo(x, 10)

These functions must be followed by a set of matching brackets:

.. code-block:: python

    tau * dmp / dt + mp = exp( - cos(2*pi*f*t + pi/4 ) + 1)

Conditional statements
____________________________

**Python-style**

It is possible to use Python-style conditional statements as the right term of an equation or ODE. They follow the form:

.. code-block:: python

    if condition : statement1 else : statement2

For example, to define a piecewise linear function, you can nest different conditionals:

.. code-block:: python

    r = if mp < 1. :
            if mp > 0.:
                mp
            else:
                0.
        else:
            1.

which is equivalent to:

.. code-block:: python

    r = clip(mp, 0.0, 1.0)

The condition can use the following vocabulary:

.. code-block:: python

    True, False, and, or, not, is, is not, ==, !=, >, <, >=, <=

.. note::

    The ``and``, ``or`` and ``not`` logical operators must be used with parentheses around their terms. Example:

    .. code-block:: python

        var = if (mp > 0) and ( (noise < 0.1) or (not(condition)) ):
                    1.0
                else:
                    0.0


    ``is`` is equivalent to ``==``, ``is not`` is equivalent to ``!=``.


When a conditional statement is split over multiple lines, the flags must be set after the last line:

.. code-block:: python

    rate = if mp < 1.0 :
              if mp < 0.0 :
                  0.0
              else:
                  mp
           else:
              1.0 : init = 0.6

An ``if a: b else:c`` statement must be exactly the right term of an equation. It is for example NOT possible to write::

    r = 1.0 + (if mp> 0.0: mp else: 0.0) + b

**Ternary operator**

The ternary operator ``ite(cond, then, else)`` (ite stands for if-then-else) is available to ease the combination of conditionals with other terms::

    r = ite(mp>0.0, mp, 0.0)
    # is exactly the same as:
    r = if mp > 0.0: mp else: 0.0

The advantage is that the conditional term is not restricted to the right term of the equation, and can be used multiple times::

    r = ite(mp > 0.0, ite(mp < 1.0, mp, 1.0), 0.0) + ite(stimulated, 1.0, 0.0)


Custom functions
-------------------

To simplify the writing of equations, custom functions can be defined either globally (usable by all neurons and synapses) or locally (only for the particular type of neuron/synapse) using the same mathematical parser.

Global functions can be defined using the ``add_function()`` method:

.. code-block:: python

    add_function('sigmoid(x) = 1.0 / (1.0 + exp(-x))')

With this declaration, ``sigmoid()`` can be used in the declaration of any variable, for example:


.. code-block:: python

    rate = sigmoid(mp)

Functions must be one-liners, i.e. they should have only one return value. They can use as many arguments as needed, but are totally unaware of the context: all the needed information should be passed as an argument.

The types of the arguments (including the return value) are by default floating-point. If other types should be used, they should be specified at the end of the definition, after the ``:`` sign, with the type of the return value first, followed by the type of all arguments separated by commas:

.. code-block:: python

    add_function('conditional_increment(c, v, t) = if v > t : c + 1 else: c : int, int, float, float')


Local functions are specific to a Neuron or Synapse class and can only be used within this context (if they have the same name as global variables, they will override them). They can be passed as a multi-line argument to the constructor of a neuron or synapse (see later):

.. code-block:: python

    functions == """
        sigmoid(x) = 1.0 / (1.0 + exp(-x))
        conditional_increment(c, v, t) = if v > t : c + 1 else: c : int, int, float, float
    """
