#===============================================================================
#
#     BoldMonitor.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2019  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
from ANNarchy.core import Global
from ANNarchy.core.Population import Population
from ANNarchy.core.Monitor import Monitor

class BoldMonitor(Monitor):
    """
    Specialized monitor for populations. Transforms the signal *variables* into a BOLD signal.

    Using the hemodynamic model as described in:

    * Friston et al. 2000: Nonlinear Responses in fMRI: The Balloon Model, Volterra Kernels, and Other Hemodynamics. NeuroImage.
    * Friston et al. 2003: Dynamic causal modelling. NeuroImage.

    Details to the equations and parameters can be found in Friston et al. (2003). The parameter/variable names were chosen in line with the naming convention of them (aside from E_0 which is rho in the article).
    """
    def __init__(self, obj, variables=[], epsilon=1.0, alpha=0.3215, kappa=0.665, gamma=0.412, E_0=0.3424, V_0=0.02, tau_0=1.0368, record_all_variables=False, period=None, period_offset=None, start=True, net_id=0):

        """
        :param obj: object to monitor. Must be a Population or PopulationView.
        :param variables: single variable name as input for the balloon model (default: []).
        :param epsilon: re-scales the provied input signal (default: 1.0)
        :param alpha: Grubb's exponent influences the outflow f_out (default: 0.3215)
        :param kappa: influences the decay of signal s (default: 0.665)
        :param gamma: rate of flow-dependent elimination (default: 0.412)
        :param E_0: resting oxygen extraction fraction (default: 0.3424)
        :param V_0: resting blood volume fraction (default: 0.02)
        :param tau_0: hemodynamic transit time (default: 1.0368)
        :param record_all_variables: if set to True, all internal state variables of the balloon model are recored (e. g. *s*, *v* and *q*). If set to False only the BOLD output will be recorded (default False).
        :param period: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).
        :param period_offset: determines the moment in ms of recording within the period (default 0). Must be smaller than **period**.
        :param start: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.
        """

        if not isinstance(obj, Population):
            Global._error("BoldMonitors can only record full populations.")

        if isinstance(variables, list) and len(variables) > 1:
            Global._error("BoldMonitors can only record one variable.")

        if start == False:
            Global._warning("BOLD monitors always record from the begin of the simulation.")

        if Global._network[net_id]['compiled']:
            # HD (28th Jan. 2019): it is a bit unhandy to force the user to use BoldMonitors differently,
            # but to generate a bold monitor code for all variables possible beforehand,
            # as we do it for normal monitors, is not a suitable approach.
            Global._error("BoldMonitors need to be instantiated before compile.")

        super(BoldMonitor, self).__init__(obj, variables, period, period_offset, start, net_id)

        self.name = "BoldMonitor"   # for report

        # Store the parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._kappa = kappa
        self._gamma = gamma
        self._E_0 = E_0
        self._V_0 = V_0
        self._tau_0 = tau_0
        self._record_all_variables = record_all_variables

        # Overwrite default monitor code
        # Attention: the equation for the balloon model should be always
        # computed, even if a period is set!
        self._specific_template = {
            'cpp': """
// BoldMonitor recording from pop%(pop_id)s (%(pop_name)s)
class BoldMonitor%(mon_id)s : public Monitor{
public:
    BoldMonitor%(mon_id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset) {
        // balloon variables
        E = std::vector<%(float_prec)s>( ranks.size(), E_0 );
        v = std::vector<%(float_prec)s>( ranks.size(), 1.0 );
        q = std::vector<%(float_prec)s>( ranks.size(), 1.0 );
        s = std::vector<%(float_prec)s>( ranks.size(), 0.0 );
        f_in = std::vector<%(float_prec)s>( ranks.size(), 1.0 );
        f_out = std::vector<%(float_prec)s>( ranks.size(), 1.0 );

        // which results should be recorded? Enabled dependent on the
        // python arguments
        record_out_signal_ = true; // set to false?
        record_all_variables = false;

    #ifdef _DEBUG
        std::cout << "BoldMonitor initialized (" << this << ") ... " << std::endl;
    #endif
    }

    ~BoldMonitor%(mon_id)s() = default;

    void record() {
        %(float_prec)s k1 = 7 * E_0;
        %(float_prec)s k2 = 2;
        %(float_prec)s k3 = 2*E_0 - 0.2;

        std::vector<%(float_prec)s> res = std::vector<%(float_prec)s>(ranks.size());
        int i = 0;
        for(auto it = ranks.begin(); it != ranks.end(); it++, i++) {
            %(float_prec)s u = pop%(pop_id)s.%(var_name)s[*it];

            E[i] = -pow(-E_0 + 1.0, 1.0/f_in[i]) + 1;
            f_out[i] = pow(v[i], 1.0/alpha);

            %(float_prec)s _v = (f_in[i] - f_out[i])/tau_0;
            %(float_prec)s _q = (E[i]*f_in[i]/E_0 - f_out[i]*q[i]/v[i])/tau_0;
            %(float_prec)s _s = epsilon*u - kappa*s[i] - gamma*(f_in[i] - 1);
            %(float_prec)s _f_in = s[i];

            v[i] += 0.001*dt*_v;
            q[i] += 0.001*dt*_q;
            s[i] += 0.001*dt*_s;
            f_in[i] += 0.001*dt*_f_in;

            res[i] = V_0*(k1*(-q[i] + 1) + k2*(-q[i]/v[i] + 1) + k3*(-v[i] + 1));
        }

        // store the result
        if ( this->record_out_signal_ && ( (t - this->offset_) %% this->period_ == this->period_offset_ ) ){
            out_signal.push_back(res);
        }

        // record intermediate variables
        if (record_all_variables) {
            rec_E.push_back(E);
            rec_f_out.push_back(f_out);
            rec_v.push_back(v);
            rec_q.push_back(q);
            rec_s.push_back(s);
            rec_f_in.push_back(f_in);
        }

        // clear interim result
        res.clear();
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // Computation Vectors
        size_in_bytes += E.capacity() * sizeof(%(float_prec)s);
        size_in_bytes += v.capacity() * sizeof(%(float_prec)s);
        size_in_bytes += q.capacity() * sizeof(%(float_prec)s);
        size_in_bytes += s.capacity() * sizeof(%(float_prec)s);
        size_in_bytes += f_in.capacity() * sizeof(%(float_prec)s);
        size_in_bytes += f_out.capacity() * sizeof(%(float_prec)s);

        // Record
        for(int i = 0; i < out_signal.size(); i++) {
            // all vectors should have the same top-level size ...
            if (record_all_variables) {
                size_in_bytes += rec_E[i].capacity() * sizeof(%(float_prec)s);
                size_in_bytes += rec_v[i].capacity() * sizeof(%(float_prec)s);
                size_in_bytes += rec_q[i].capacity() * sizeof(%(float_prec)s);
                size_in_bytes += rec_s[i].capacity() * sizeof(%(float_prec)s);
                size_in_bytes += rec_f_in[i].capacity() * sizeof(%(float_prec)s);
                size_in_bytes += rec_f_out[i].capacity() * sizeof(%(float_prec)s);
            }
            size_in_bytes += out_signal[i].capacity() * sizeof(%(float_prec)s);
        }

        return size_in_bytes;
    }

    void clear() {
        //std::cout << "BoldMonitor::clear (" << this << ") ... " << std::endl;

        /* Clear state data */
        E.clear();
        E.shrink_to_fit();
        v.clear();
        v.shrink_to_fit();
        q.clear();
        q.shrink_to_fit();
        s.clear();
        s.shrink_to_fit();
        f_in.clear();
        f_in.shrink_to_fit();

        /* Clear recorded data, first sub arrays
           then top-level
         */
        out_signal.clear();
        out_signal.shrink_to_fit();
        rec_E.clear();
        rec_E.shrink_to_fit();
        rec_f_out.clear();
        rec_f_out.shrink_to_fit();
        rec_v.clear();
        rec_v.shrink_to_fit();
        rec_q.clear();
        rec_q.shrink_to_fit();
        rec_s.clear();
        rec_s.shrink_to_fit();
        rec_f_in.clear();
        rec_f_in.shrink_to_fit();
    }

    void record_targets() {} // nothing to do here ...

    void set_record_all_variables(bool value) {
    #ifdef _DEBUG
        if (record_all_variables)
            std::cout << "Recording all state variables enabled for BoldMonitor pop%(pop_id)s";
    #endif
        record_all_variables = value;
    }

    bool get_record_all_variables() {
        return record_all_variables;
    }

    std::vector< std::vector<%(float_prec)s> > out_signal;
    // record intermediate variables
    std::vector< std::vector<%(float_prec)s> > rec_E;
    std::vector< std::vector<%(float_prec)s> > rec_f_out;
    std::vector< std::vector<%(float_prec)s> > rec_v;
    std::vector< std::vector<%(float_prec)s> > rec_q;
    std::vector< std::vector<%(float_prec)s> > rec_s;
    std::vector< std::vector<%(float_prec)s> > rec_f_in;

    %(float_prec)s epsilon;
    %(float_prec)s alpha;
    %(float_prec)s kappa;
    %(float_prec)s gamma;
    %(float_prec)s E_0;
    %(float_prec)s V_0;
    %(float_prec)s tau_0;

    bool record_out_signal_;

private:
    bool record_all_variables; // Enabling only for debug purposes!
    %(float_prec)s k1_;
    %(float_prec)s k2_;
    %(float_prec)s k3_;

    std::vector<%(float_prec)s> E;
    std::vector<%(float_prec)s> v;
    std::vector<%(float_prec)s> q;
    std::vector<%(float_prec)s> s;
    std::vector<%(float_prec)s> f_in;
    std::vector<%(float_prec)s> f_out;
};
""",
            'pyx_struct': """

    # BoldMonitor%(mon_id)s recording from %(pop_id)s (%(pop_name)s)
    cdef cppclass BoldMonitor%(mon_id)s (Monitor):
        BoldMonitor%(mon_id)s(vector[int], int, int, long) except +

        long int size_in_bytes()
        void clear()

        bool get_record_all_variables()
        void set_record_all_variables(bool value)

        # record BOLD output
        vector[vector[%(float_prec)s]] out_signal

        # record intermediate variables
        vector[vector[%(float_prec)s]] rec_E
        vector[vector[%(float_prec)s]] rec_f_out
        vector[vector[%(float_prec)s]] rec_v
        vector[vector[%(float_prec)s]] rec_q
        vector[vector[%(float_prec)s]] rec_s
        vector[vector[%(float_prec)s]] rec_f_in

        %(float_prec)s epsilon
        %(float_prec)s alpha
        %(float_prec)s kappa
        %(float_prec)s gamma
        %(float_prec)s E_0
        %(float_prec)s V_0
        %(float_prec)s tau_0

""",
            'pyx_wrapper': """

# Population Monitor wrapper
cdef class BoldMonitor%(mon_id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new BoldMonitor%(mon_id)s(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<BoldMonitor%(mon_id)s *>self.thisptr).size_in_bytes()

    def clear(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).clear()

    property record_all_variables:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).get_record_all_variables()
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).set_record_all_variables(val)

    # Output
    property out_signal:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).out_signal
    def clear_out_signal(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).out_signal.clear()

    # Intermediate Variables
    property E:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_E
    def clear_E(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_E.clear()
    property f_out:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_f_out
    def clear_f_out(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_f_out.clear()
    property v:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_v
    def clear_v(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_v.clear()
    property q:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_q
    def clear_q(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_q.clear()
    property s:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_s
    def clear_s(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_s.clear()
    property f_in:
        def __get__(self): return (<BoldMonitor%(mon_id)s *>self.thisptr).rec_f_in
    def clear_f_in(self):
        (<BoldMonitor%(mon_id)s *>self.thisptr).rec_f_in.clear()

    # Parameters
    property epsilon:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).epsilon = val
    property alpha:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).alpha = val
    property kappa:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).kappa = val
    property gamma:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).gamma = val
    property E_0:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).E_0 = val
    property V_0:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).V_0 = val
    property tau_0:
        def __set__(self, val): (<BoldMonitor%(mon_id)s *>self.thisptr).tau_0 = val

"""
        }

    #######################################
    ### Attributes
    #######################################
    # Record intermediate variables
    @property
    def record_all_variables(self):
        if not self.cyInstance:
            return self._record_all_variables
        else:
            return self.cyInstance.record_all_variables
    @record_all_variables.setter
    def record_all_variables(self, val):
        if not self.cyInstance:
            self._record_all_variables = val
        else:
            self.cyInstance.record_all_variables = val

    # epsilon
    @property
    def epsilon(self):
        "TODO"
        if not self.cyInstance:
            return self._epsilon
        else:
            return self.cyInstance.epsilon
    @epsilon.setter
    def epsilon(self, val):
        if not self.cyInstance:
            self._epsilon = val
        else:
            self.cyInstance.epsilon = val

    # alpha
    @property
    def alpha(self):
        "TODO"
        if not self.cyInstance:
            return self._alpha
        else:
            return self.cyInstance.alpha
    @alpha.setter
    def alpha(self, val):
        if not self.cyInstance:
            self._alpha = val
        else:
            self.cyInstance.alpha = val
    # kappa
    @property
    def kappa(self):
        "TODO"
        if not self.cyInstance:
            return self._kappa
        else:
            return self.cyInstance.kappa
    @kappa.setter
    def kappa(self, val):
        if not self.cyInstance:
            self._kappa = val
        else:
            self.cyInstance.kappa = val
    # gamma
    @property
    def gamma(self):
        "TODO"
        if not self.cyInstance:
            return self._gamma
        else:
            return self.cyInstance.gamma
    @gamma.setter
    def gamma(self, val):
        if not self.cyInstance:
            self._gamma = val
        else:
            self.cyInstance.gamma = val
    # E_0
    @property
    def E_0(self):
        "TODO"
        if not self.cyInstance:
            return self._E_0
        else:
            return self.cyInstance.E_0
    @E_0.setter
    def E_0(self, val):
        if not self.cyInstance:
            self._E_0 = val
        else:
            self.cyInstance.E_0 = val
    # V_0
    @property
    def V_0(self):
        "TODO"
        if not self.cyInstance:
            return self._V_0
        else:
            return self.cyInstance.V_0
    @V_0.setter
    def V_0(self, val):
        if not self.cyInstance:
            self._V_0 = val
        else:
            self.cyInstance.V_0 = val
    # tau_0
    @property
    def tau_0(self):
        "TODO"
        if not self.cyInstance:
            return self._tau_0
        else:
            return self.cyInstance.tau_0
    @tau_0.setter
    def tau_0(self, val):
        if not self.cyInstance:
            self._tau_0 = val
        else:
            self.cyInstance.tau_0 = val

    # Intermediate Variables
    @property
    def E(self):
        return self.cyInstance.E

    @property
    def f_in(self):
        return self.cyInstance.f_in

    @property
    def f_out(self):
        return self.cyInstance.f_out

    @property
    def v(self):
        return self.cyInstance.v

    @property
    def q(self):
        return self.cyInstance.q

    @property
    def s(self):
        return self.cyInstance.s

    #######################################
    ### Data access
    #######################################
    def _start_bold_monitor(self):
        """
        Automatically called from Compiler._instantiate()
        """
        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'BoldMonitor'+str(self.id)+'_wrapper')(self.object.ranks, period, period_offset, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        # Set the parameter
        self.cyInstance.epsilon = self._epsilon
        self.cyInstance.alpha = self._alpha
        self.cyInstance.kappa = self._kappa
        self.cyInstance.gamma = self._gamma
        self.cyInstance.E_0 = self._E_0
        self.cyInstance.V_0 = self._V_0
        self.cyInstance.tau_0 = self._tau_0
        self.record_all_variables = self._record_all_variables

    def get(self, variables=None, keep=False):
        """
        Get the recorded BOLD signal.
        """
        if variables == None:
            return self._get_population(self.object, name="out_signal", keep=keep)
        else:
            raise ValueError
