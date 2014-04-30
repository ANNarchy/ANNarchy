**************************
Parser
**************************

A Neuron or Synapse type is primarily defined by two sets of values which must be specified in its constructor:
    
* **Parameters** are values such as time constants which are constant during the simulation. They can be the same throughout the population, or different.

* **Variables** are neuronal variables (for example the membrane potential or firing rate) whose value evolve with time. The equation (differential or not) ruling their evolution can be described using a specific meta-language.  


Parameters
---------------------

Parameters are defined by a multi-string consisting of one or more parameter definitions:

.. code-block:: python

    parameters = """
        tau = 10.0
        eta = 0.5
    """

Each parameter should be defined on a single line, with its name on the left side of the equal sign, and its value on the right side. The given value corresponds to the  initial value of the parameter (but it can be overloaded before the start of the simulation).


Constraints
____________

By default, a neural parameter will be unique to each neuron (i.e. each neuron instance will hold a copy of the parameter). In order to save memory usage, one can force ANNarchy to store one parameter value for a whole population by specifying the ``population`` flag after a ``:`` symbol following the parameter definition:

.. code-block:: python

    parameters = """
        tau = 10.0
        eta = 0.5 : population
    """
    
The same is true for synapses, whose parameters are unique to each synapse in a given projection. If the ``postsynaptic`` flag is passed, the parameter will be common to all synapses of a postsynaptic neuron. 
    
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
        tau * dmp/dt  = baseline - mp + sum(exc) + noise
        rate = pos(mp)
    """

The evolution of each variable with time can be described through a simple equation or an ordinary differential equation (ODE). ANNarchy provides a simple parser for mathematical expressions, whose role is to translate a high-level description of the equation into an optimized C++ code portion.

The equation for one variable can depend on parameters, other variables (even when declared later) or numerical constants. variables are updated in the same order as their declaration in the multistring.

The declaration of a variable can extend on multiple lines:

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

In practice, ODEs are transformed using Sympy into the last form (only the gradient stays on the left) and numerized using either forward (the default), semi-implicit (or backward) or exponential Euler methods.


Constraints
____________

Variables also accept the ``population``, ``postsynaptic``, ``int`` or ``bool`` flags. In addition, the initial value of the variable (before the first simulation starts) can also be specified using the ``init`` keyword followed by the desired value:


.. code-block:: python

    equations="""    
        tau * dmp / dt  = baseline - mp : init = 0.2
    """

Upper- and lower-bounds can also be set using the ``min`` and ``max`` keywords:

.. code-block:: python

    equations="""    
        tau * dmp / dt  = baseline - mp : min = -0.2, max = 1.0
    """
        
The numerization method for ODEs can be explicitely set by the ``exponential`` and ``implicit`` keywords (when omitted, the default is the forward Euler method)

.. code-block:: python

    equations="""    
        tau * dmp / dt  = baseline - mp : implicit
    """

**Summary of allowed keywords for variables:**

* *init*: defines the initialization value at begin of simulation and after a network reset (default: 0.0)
* *min*: minimum allowed value  (unset by default)
* *max*: maximum allowed value (unset by default)
* *population*: the attribute is equal for all neurons in a population.
* *postsynaptic*: the attribute is equal for all synapses of a postsynaptic neuron.
* *exponential*: the linear ODE will be integrated using the exponential Euler method.
* *implicit*: the ODE will be integrated using the semi-implicit (or backward) Euler method. 

Allowed vocabulary
___________________

The mathematical parser relies heavily on the one provided by `SymPy <http://sympy.org/>`_. 

**Constants**

All parameters and variables use implicitely the floating-point double precision, except when stated otherwise with the ``int`` or ``bool`` keywords. You can use numerical constants within the equation, noting that they will be automatically converted to this precision:

.. code-block:: python

    tau * dmp / dt  = 1 / pos(mp) + 1 
    
The constant :math:`\pi` is available under the literal form ``pi``.
    
**Operators**

* Additions (+), substractions (-), multiplications (*), divisions (/) and power functions (^) are of course allowed.

* Gradients are allowed only for the variable currently described. They take the form:

.. code-block:: python

    dmp / dt  = A
    
with a ``d`` preceding the variable's name and terminated by ``/dt`` (with or without spaces). Gradients must be on the left side of the equation.

* To update the value of a variable at each time step, the operators ``=``, ``+=``, ``-=``, ``*=``, and ``/=`` are allowed.

**Random number generators**

Several random generators are available and can be used within an equation. In the current version are available:

* ``Uniform(min, max)`` generates random numbers from a uniform distribution in the range :math:`[\text{min}, \text{max}]`.

* ``Normal(mu, sigma)`` generates random numbers from a normal distribution with min mu and standard deviation sigma.

Example:

.. code-block:: python

    noise = Uniform(-0.5, 0.5)

**Mathematical functions**

Most mathematical functions of the ``cmath`` library are understood by the parser, for example:

.. code-block:: python

    cos, sin, tan, acos, asin, atan, exp, abs, fabs, sqrt, log, ln
    
The positive and negative parts of a term are also defined:

.. code-block:: python
    
    pos, positive, neg,  negative
    
These functions must be followed by a set of matching brackets:

.. code-block:: python

    tau * dmp / dt + mp = exp( - cos(2*pi*f*t + pi/4 ) + 1)
    
**Conditional statements**

It is possible to use conditional statements inside an equation or ODE. They follow the form:

.. code-block:: python

    if condition : statement1 else : statement2
    
For example, to define a piecewise linear function, you can write: 

.. code-block:: python

    rate = if mp < 1 : pos(mp) else: 1
 
The condition can use the following vocabulary:

.. code-block:: python
    
    True, False, and, or, not, >, <, >=, <=
    
.. note::

    The ``and``, ``or`` and ``not`` logical operators must be used with parentheses around their terms. Example:
    
    .. code-block:: python
    
        weirdo = if (mp > 0) and ( (noise < 0.1) or (not(condition)) ): 1.0 else: 0.0     
    
Conditional statements can also be nested:

.. code-block:: python

    rate = if mp < 1.0 : 
              if mp < 0.0 :
                  0.0 
              else: 
                  mp
           else:
              1.0

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


