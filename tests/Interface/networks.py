from ANNarchy import add_function, clear, Constant, Hebb, IF_curr_exp, \
    load_parameters, Monitor, Network, Neuron, STDP, Synapse, Population, \
    Projection, save_parameters, Uniform

def define_rate_net():
    Constant('mu', 0.2)
    add_function("fa(x) = 2/(x*x)")
    DefaultNeuron = Neuron(
        parameters = """
            tau = 10.0 : population
            baseline = 1.0 : population
            condition = True
        """,
        equations= """
            noise = Uniform(-0.1, 0.1)
            tau * dmp/dt + mp = fa(sum(exc)) +  neg(- fb(sum(inh)))
                                + baseline + noise : implicit
            r = pos(mp)
        """,
        functions="""
            fb(x) = if x>0: x else:x/2
        """
    )

    emptyNeuron1 = Neuron(equations="r=0")
    emptyNeuron2 = Neuron(parameters="r=0")

    Oja = Synapse(
        parameters = """
            eta = 10.0
            tau = 10.0 : projection
        """,
        equations = """
            tau * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
            eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
            mu * dx/dt = pre.sum(exc) * post.sum(exc)
        """,
        psp = """
            w * pre.r
        """
    )

    pop1 = Population(name='pop1', neuron=emptyNeuron1, geometry=1)
    pop2 = Population(name='pop2', neuron=DefaultNeuron, geometry=1)
    pop3 = Population(name='pop3', neuron=emptyNeuron2, geometry=(2,2))
    proj1 = Projection(pre=pop1, post=pop2, target='exc')
    proj1.connect_one_to_one(weights=Uniform(-0.5, 0.5))
    proj2 = Projection(pre=pop1, post=pop3, target='exc', synapse=Hebb)
    proj2.connect_all_to_all(1.0)
    proj3 = Projection(pre=pop2, post=pop3, target='exc', synapse=Oja)
    proj3.connect_all_to_all(1.0)

    m = Monitor(pop2,'r')

    network = Network()
    network.add([pop1, pop2, pop3, proj1, proj2, proj3, m])
    network.compile(silent=True)

    return network, network.get(pop2), network.get(proj2), network.get(proj3)

def define_spike_net():
    Constant('b', 0.2)
    add_function("f(x) = 5.0 * x + 140")
    Izhikevich = Neuron(
        parameters = """
            noise = 0.0
            a = 0.02 : population
            c = -65.0
            d = 8.0
            v_thresh = 30.0
            i_offset = 0.0
        """,
        equations = """
            I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset
            dv/dt = f2(v) + f(v) - u + I : init = -65.0
                du/dt = a * (b*v - u) : init= -13.0
        """,
        spike = "v > v_thresh",
        reset = "v = c; u += d",
        refractory = 0.0,
        functions="f2(x) = 0.04 * x^2"
    )

    STP = Synapse(
        parameters = """
            tau_rec = 100.0 : projection
            tau_facil = 0.01 : projection
            U = 0.5
        """,
        equations = """
            dx/dt = (1 - x)/tau_rec : init = 1.0, event-driven
            du/dt = (U - u)/tau_facil : init = 0.5, event-driven
        """,
        pre_spike="""
            g_target += w * u * x
            x *= (1 - u)
            u += U * (1 - u)
            y = pre.u + post.I
        """,
        post_spike="y += post.I",
    )

    pop1 = Population(name='pop1', neuron=IF_curr_exp, geometry=1)
    pop2 = Population(name='pop2', neuron=Izhikevich, geometry=1)
    pop3 = Population(name='pop3', neuron=Izhikevich, geometry=(2,2))
    proj1 = Projection(pre=pop1, post=pop2, target='exc',
                            synapse=STDP)
    proj1.connect_one_to_one(weights=Uniform(-0.5, 0.5))
    proj2 = Projection(pre=pop2, post=pop3, target='exc', synapse=STP)
    proj2.connect_all_to_all(1.0)

    m = Monitor(pop2,'r')

    network = Network()
    network.add([pop1, pop2, pop3, proj1, proj2, m])
    network.compile(silent=True)

    return network, network.get(pop2), network.get(proj1), network.get(proj2)
