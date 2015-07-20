***********************************
Equations and numerical methods
***********************************

Numerical methods
*****************

First-order ordinary differential equations (ODE) can be solved using different numerical methods. The method can be declared globally in the ``setup()`` call and used in all ODEs of the network::

    from ANNarchy import *

    setup(method='exponential')

or specified explicitely for each ODE by specifying a flag:

.. code-block:: python

    equations = """    
        tau * dV/dt  + V =  A : init = 0.0, exponential
    """

If nothing is specified, the explicit Euler method will be used.

Different numerical methods are available: 

* Explicit Euler ``'explicit'``
* Implicit Euler ``'implicit'``
* Exponential Euler ``'exponential'``
* Midpoint ``'midpoint'``
* Event-driven ``'event-driven'``
  
Each method has advantages/drawbacks in term of numerical error, stability and computational cost.  

To describe these methods, we will take the example of a system of two linear first-order ODEs:

.. math::

    \frac{dx(t)}{dt} = f(x(t), y(t)) = a_x \cdot x(t) + b_x \cdot y(t) + c_x

    \frac{dy(t)}{dt} = g(x(t), y(t)) = a_y \cdot x(t) + b_y \cdot y(t) + c_y

The objective of a numerical method is to approximate the value of :math:`x` and :math:`y` at time :math:`t+h` based on its value at time :math:`t`, where :math:`h` is the discretization time step (noted ``dt`` in ANNarchy):


.. math::

    x(t + h) = F(x(t), y(t)) 

    y(t + h) = G(x(t), y(t))

At each step of the simulation, the new values for the variables are computed using this update rule and will be used for the following step. 

The derivative of each variable is usually approximated by:

.. math::

    \frac{dx(t)}{dt} = \frac{x(t+h) - x(t)}{h}

    \frac{dy(t)}{dt} = \frac{y(t+h) - y(t)}{h}

The different numerical methods mostly differ in the time at which the functions :math:`f` and :math:`g` are evaluated.


Explicit Euler method
=====================


The explicit (forward) Euler method computes the next value for the variables by estimating their derivative at time :math:`t`:

.. math::

    \frac{dx(t)}{dt} = \frac{x(t+h) - x(t)}{h} = f(x(t), y(t)) 

    \frac{dy(t)}{dt} = \frac{y(t+h) - y(t)}{h} = g(x(t), y(t))

so the solution is straightforward to obtain: 

.. math::

    x(t+h) =  x(t) + h \cdot  f(x(t), y(t)) 

    y(t+h) = y(t) + h \cdot g(x(t), y(t))


Implicit Euler method
=====================

The explicit (forward) Euler method computes the next value for the variables by estimating their derivative at time :math:`t + h`:

.. math::

    \frac{dx(t)}{dt} = \frac{x(t+h) - x(t)}{h} = f(x(t+h), y(t+h)) 

    \frac{dy(t)}{dt} = \frac{y(t+h) - y(t)}{h} = g(x(t+h), y(t+h))

This leads to a system of equations which must be solved in order to find the update rule. With the linear equations defined above, we need to solve: 

.. math::

    \frac{x(t+h) - x(t)}{h} = a_x \cdot x(t + h) + b_x \cdot y(t + h) + c_x

    \frac{y(t+h) - y(t)}{h} = a_y \cdot x(t + h) + b_y \cdot y(t + h) + c_y

what gives something like: 

.. math::

    x(t+h) =  x(t) - h \cdot \frac{ \left(a_{x} x(t) + b_{x} y(t) + c_{x} + h \left(- a_{x} b_{y} x(t) + a_{y} b_{x} x(t) + b_{x} c_{y} - b_{y} c_{x}\right)\right)}{h^{2} \left(- a_{x} b_{y} + a_{y} b_{x}\right) + h \left(a_{x} + b_{y}\right) - 1}

    y(t+h) = y(t) -h \cdot  \frac{ a_{y} \left(c_{x} h + x(t)\right) + y(t) \left(- a_{y} b_{x} h^{2} + \left(a_{x} h - 1\right) \left(b_{y} h - 1\right)\right) + \left(a_{x} h - 1\right) \left(c_{y} h + y(t)\right)}{a_{y} b_{x} h^{2} - \left(a_{x} h - 1\right) \left(b_{y} h - 1\right)} 



ANNarchy relies on Sympy to solve this system of equations and generate the update rule.


Exponential Euler
=================

The exponential Euler method is particularly stable for single first-order linear equations, of the type:


.. math::

    \tau(t) \cdot \frac{dx(t)}{dt}  + x(t) =  A(t)



The update rule is then given by: 

.. math::

    x(t+h) = x(t) + (1 - \exp(- \frac{h}{\tau(t)}) ) \cdot (A(t) - x(t))


The difference with the explicit Euler method is the step size, which is an exponential function of the ratio :math:`\frac{\tau}{h}`. The accurary of the exponential Euler method on linear first-order ODEs is close to perfect, compared to the other Euler methods. As it is an explicit method, systems of equations are solved very easily with the same rule. 


When the exponential method is used, ANNarchy first tries to reduce the ODE to its canonical form above (with the time constant being possibly dependent on time or inputs) and then generates the update rule accordingly. 

For example, the description::

    tau * dv/dt = (E - v) + g_exc * (Ee - v) + g_inh * (v - Ei)

would be first transformed in::

    (1 + g_exc - g_inh) * dv/dt + v = (E + g_exc * Ee - g_inh * Ei) / (1 + g_exc - g_inh)

before being transformed into an update rule, with :math:`\tau(t) = 1 + g_\text{exc} - g_\text{inh}`:


.. math::

    v(t+h) = v(t) + (1 - \exp(- \frac{h}{1 + g_\text{exc} - g_\text{inh}}) ) \cdot (\frac{E + g_\text{exc} \cdot E_e - g_\text{inh} \cdot E_i}{1 + g_\text{exc} - g_\text{inh}} - v(t))


.. warning::

    The exponential method can only be applied to **first-order linear** ODEs. Any other form of ODE will be rejected by the parser.


Midpoint
=========

The midpoint method is a Runge-Kutta method of order 2. It estimates the derivative in the middle of the interval :math:`t + \frac{h}{2}`.


.. math::

    k_x = f(x(t), y(t)) 

    k_y = g(x(t), y(t))

    x(t+h) =  x(t) + h \cdot  f(x(t) + k_x \cdot \frac{h}{2}, y(t) +  k_y \cdot \frac{h}{2}) 

    y(t+h) = y(t) + h \cdot g(x(t) + k_x \cdot \frac{h}{2}, y(t) +  k_y \cdot \frac{h}{2})


Event-driven
=============

Event-driven integration is only available for spiking synapses with variables following linear first-order dynamics. Let's consider the following STDP synapse (see :doc:`SpikeSynapse` for explanations)::

    STDP = Synapse(
        parameters = """
            tau_pre = 10.0 : post-synaptic
            tau_post = 10.0 : post-synaptic
        """,
        equations = """
            tau_pre * dApre/dt = - Apre : event-driven
            tau_post * dApost/dt = - Apost : event-driven
        """,
        pre_spike = """
            g_target += w
            Apre += cApre 
            w = clip(w + Apost, 0.0 , 1.0)
        """,                  
        post_spike = """
            Apost += cApost
            w = clip(w + Apre, 0.0 , 1.0)
        """      
    ) 

The value of ``Apost`` and ``Apre`` is only needed when a pre- or post-synaptic spike occurs at the synapse, so there is no need to integrate the corresponding equations between two such events. First-order linear ODEs have the nice property that their analytical solution is easy to obtain. Let's consider an equation of the form:

.. math::

    \tau  \frac{dv}{dt} = E - v

If :math:`v` has the value :math:`V_0` at time :math:`t`,, its value at time :math:`t + \Delta t` is given by:

.. math::

    v(t + \Delta t) = V_0 \cdot \exp(-\frac{\Delta t}\tau}) 


.. note::

    If the synapse defines a ``psp`` argument (synaptic transmission is continuous), it is not possible to use event-driven integration.


Order of evaluation
**********************

The values of variables are stored in a single array in order to save some memory. Special care therefore has to be taken on whether the update of a variable depends on the value of another variable at the previous time step or in the same step. 

Systems of ODEs
===============

Systems of ODEs are integrated concurrently, which means that the following system::

    tau*dv/dt = I - v - u 
    tau*du/dt = v - u

would be numerized using the explicit Euler method as::

    v[t+1] = v[t] + dt*(I - v[t] - u[t])/tau
    u[t+1] = u[t] + dt*(v[t] - u[t])/tau


As we use a single array, the generated code is similar to::

    new_v = v + dt*(I - v - u)/tau
    new_u = u + dt*(v - u)/tau

    v = new_v
    u = new_u

This way, we ensure that the interdependent ODEs use the correct value for the other variables.

Assignments
============

When assignments (``=``, ``+=``...) are used in an ``equations`` field, the order of valuation is different:

* Assigments occurring before or after a system of ODEs are updated sequentially.
* Systems of ODEs are updated concurrently.

Let's consider the following dummy equations::

    # Process the inputs
    Exc = some_function(sum(exc))
    Inh = another_function(sum(inh))
    I = Exc - Inh
    # ODE for the membrane potential, with a recovery variable
    tau*dv/dt = I - v - u
    tau*du/dt = v - u
    # Firing rate is the positive part of v
    r = pos(v)

Here,  ``Exc`` and ``Inh`` represent the inputs to the neuron at the current time ``t``. The new values should be immediately available for updating ``I``, whose value should similarly be immediately used in the ODE of ``v``. Similarly, the value of ``r`` should be the positive part of the value of ``v`` that was just calculated, not at the previous time step. Doing otherwise would introduce a lag in the neuron: changes in ``sum(exc)`` at ``t`` would be reflected in ``Exc`` at ``t+1``, in ``I`` at ``t+2``, in ``v`` at ``t+3`` and finally in ``r`` at ``t+4``. This is generally unwanted.

The generated code is therefore equivalent to::

    # Process the inputs
    Exc = some_function(sum(exc))
    Inh = another_function(sum(inh))
    I = Exc - Inh
    # ODE for the membrane potential, with a recovery variable
    new_v = v + dt*(I - v - u)/tau
    new_u = u + dt*(v - u)/tau
    v = new_v
    u = new_u
    # Firing rate is the positive part of v
    r = pos(v)


One can even define multiple groups of assignments and systems of ODEs: systems of ODEs separated by at least one assignment will be evaluated sequentially (but concurrently inside each system). For example, in::

    tau*du/dt = v - u
    I = g_exc - g_inh
    tau*dk/dt = v - k
    tau*dv/dt = I - v - u + k

``u`` and ``k`` are updated using the previous value of ``v``, while ``v`` uses the new values of both ``I`` and ``u``, but the previous one of ``k``.
