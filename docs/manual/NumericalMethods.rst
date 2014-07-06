**************************
Numerical methods
**************************

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
* Semi-implicit Euler ``'semiimplicit'``
* Exponential Euler ``'exponential'``
* Midpoint ``'midpoint'``
  
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


Semi-implicit Euler method
==========================

If the ODEs are not linear, the Sympy solver may not come to a satisfyable solution for the implicit method: some non-linear functions may not be inversible, the uniqueness of the solution may not be guaranteed, some inverse functions have no equivalent in the C math library, etc.

In this case, ANNarchy will switch automatically to the semi-implicit (or hybrid) method, as defined in:

    **Izhikevich E. M.** (2010). Hybrid Spiking Models. *Phil. Trans. R. Soc. A 368:5061-5070*

Let's consider a single quadratic ODE: 

.. math::

    \frac{dx(t)}{dt} = a \cdot x(t)^2 + b \cdot x(t) + c 

Using the implicit method would require to solve a quadratic polynome of :math:`x(t+h)`, which generally admits two solutions. ANNarchy can not know which one is the correct one.

The semi-implicit method will estimate the derivative at time :math:`t + h` for the linear part of the ODE, but at time :math:`t` for the non-linear part:

.. math::

    \frac{x(t+h) - x(t)}{h} = a \cdot x(t)^2 + b \cdot x(t + h) + c

so we obtain a much simpler although stable update rule: 


.. math::

    x(t+h) = x(t) + \frac{h}{1 - h \cdot b} \cdot ( a \cdot x(t)^2 + b \cdot x(t) + c )

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

