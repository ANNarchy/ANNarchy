# =============================================================================
#
#     CopyProjection.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Julien Vitay <julien.vitay@gmail.com>,
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
# =============================================================================
from ANNarchy.core import Global
from ANNarchy.core.Projection import Projection
from ANNarchy.models.Synapses import DefaultRateCodedSynapse, DefaultSpikingSynapse

class Transpose(Projection):
    """
    Creates a virtual inverted projection reusing the weights and delays of an already-defined projection.

    Even though the original projection can be learnable, this one can not.
    """
    def __init__(self, proj, target):
        """
        :param proj: original projection
        :param target: type of the connection (can differ from the original one)
        """
        default_synapse = DefaultRateCodedSynapse if proj.pre.neuron_type == "rate" else DefaultSpikingSynapse

        Projection.__init__(
            self,
            pre = proj.post,
            post = proj.pre,
            target = target,
            synapse = default_synapse
        )

        self.fwd_proj = proj

        # simply copy from the forward view
        self.delays = proj._connection_delay
        self.max_delay = proj.max_delay
        self.uniform_delay = proj.uniform_delay

    def _copy(self):
        raise NotImplementedError

    def _create(self):
        print("xxx")

    def _connect(self, module):
        # create fake LIL object to have the forward view in C++
        try:
            from ANNarchy.core.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Global._print(e)
            Global._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.max_delay
        lil.uniform_delay = self.uniform_delay
        self.connector_name = "Transpose"
        self.connector_description = "Transpose"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays)

    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """
        from ANNarchy.generator.Projection import LIL_OpenMP
        # remove forward view
        self._specific_template['declare_connectivity_matrix'] = ""
        self._specific_template['access_connectivity_matrix'] = ""
        
        # remove monitor
        self._specific_template['monitor_class'] = ""
        self._specific_template['pyx_wrapper'] = ""

        self._specific_template['psp_code'] = "" 
        """ LIL_OpenMP.lil_summation_operation['sum'] % {
            'pre_copy': '',
            'omp_code': '',
            'psp': 'w[i][j] * pop%(id_pre)s.r[i]' % {'id_pre':self.pre.id},
            'id_post': self.post.id,
            'target': self.target,
            'post_index': '[post_rank[i]]'
        }"""
        print(self._specific_template['psp_code'])
        