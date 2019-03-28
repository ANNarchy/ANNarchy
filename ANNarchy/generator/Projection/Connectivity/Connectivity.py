#===============================================================================
#
#     Connectivity.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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

# list of list
from ANNarchy.generator.Projection.Connectivity import LIL_CUDA
from ANNarchy.generator.Projection.Connectivity import LIL_OpenMP

# compressed sparse row (pre1st)
from ANNarchy.generator.Projection.Connectivity import CSR_CUDA
from ANNarchy.generator.Projection.Connectivity import CSR_OpenMP

class Connectivity(object):
    """
    Base class to define connectivities in ANNarchy, the derived classes are
    responsible to select the correct templates and store them in the variable
    self._templates
    """
    def __init__(self):
        """
        Constructor does not initialize anything. Child classes are responsible to
        initialize necessary data. Secondly the method configure() should be called
        before interact with this class.
        """
        super(Connectivity, self).__init__()

    def configure(self, proj):
        """
        This function should be called before any other method of the
        Connectivity class is called, as the templates are adapted.
        """
        raise NotImplementedError

    def _connectivity(self, proj):
        """
        Create codes for connectivity, comprising usually of post_rank and
        pre_rank. In case of spiking models they are extended by an inv_rank
        data field. The extension SharedProjection as well as
        SpecificProjection members overwrite the _specific_template member
        variable of the Projection object, to replace directly the default
        codes.

        Returns:

            a dictionary containing the following fields: *declare*, *init*,
            *accessor*, *declare_inverse*, *init_inverse*

        TODO:

            Some templates require additional information (e. g. pre- or post-
            synaptic population id) and some not. Yet, I simply add the informa-
            tion in all cases, even if there are not absolutely necessary.
        """
        declare_inverse_connectivity_matrix = ""
        init_inverse_connectivity_matrix = ""

        # Retrieve the templates
        connectivity_matrix_tpl = self._templates['connectivity_matrix']

        # Special case if there is a constant weight across all synapses
        if proj._has_single_weight():
            weight_matrix_tpl = self._templates['single_weight_matrix']
        else:
            weight_matrix_tpl = self._templates['weight_matrix']

        # Connectivity
        declare_connectivity_matrix = connectivity_matrix_tpl['declare']
        access_connectivity_matrix = connectivity_matrix_tpl['accessor']
        init_connectivity_matrix = connectivity_matrix_tpl['init'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'target': proj.target
        }

        # Weight array
        declare_connectivity_matrix += weight_matrix_tpl['declare'] % {'float_prec': Global.config['precision']}
        access_connectivity_matrix += weight_matrix_tpl['accessor'] % {'float_prec': Global.config['precision']}
        init_connectivity_matrix += weight_matrix_tpl['init']

        # Spiking model require inverted ranks
        if proj.synapse_type.type == "spike":
            inv_connectivity_matrix_tpl = self._templates['inverse_connectivity_matrix']
            declare_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['declare']
            init_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['init']

        # Specific projections can overwrite
        if 'declare_connectivity_matrix' in proj._specific_template.keys():
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix']
        if 'access_connectivity_matrix' in proj._specific_template.keys():
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']
        if 'declare_inverse_connectivity_matrix' in proj._specific_template.keys():
            declare_inverse_connectivity_matrix = proj._specific_template['declare_inverse_connectivity_matrix']
        if 'init_connectivity_matrix' in proj._specific_template.keys():
            init_connectivity_matrix = proj._specific_template['init_connectivity_matrix']
        if 'init_inverse_connectivity_matrix' in proj._specific_template.keys():
            init_inverse_connectivity_matrix = proj._specific_template['init_inverse_connectivity_matrix']

        return {
            'declare' : declare_connectivity_matrix,
            'init' : init_connectivity_matrix,
            'accessor' : access_connectivity_matrix,
            'declare_inverse': declare_inverse_connectivity_matrix,
            'init_inverse': init_inverse_connectivity_matrix
        }

class OpenMPConnectivity(Connectivity):
    """
    Implementor class to define connectivities in ANNarchy for single- and multi-core CPUs.
    """
    def __init__(self):
        """
        Initialization of the class and updates the _templates variable of
        ProjectionGenerator with some fields.
        """
        super(OpenMPConnectivity, self).__init__()

    def configure(self, proj):
        """
        Assign the correct template dictionary based on projection
        storage format.
        """
        if proj._storage_format == "lil":
            self._templates.update(LIL_OpenMP.conn_templates)
        elif proj._storage_format == "csr":
            self._templates.update(CSR_OpenMP.conn_templates)
        else:
            raise NotImplementedError

class CUDAConnectivity(Connectivity):
    """
    Implementor class to define connectivities in ANNarchy for Nvidia CUDA.
    """
    def __init__(self):
        """
        Initialization of the class and updates the _templates variable of
        ProjectionGenerator with some fields.
        """
        super(CUDAConnectivity, self).__init__()

    def configure(self, proj):
        """
        Assign the correct template dictionary based on projection
        storage format.
        """
        if proj._storage_format == "lil":
            self._templates.update(LIL_CUDA.conn_templates)
        elif proj._storage_format == "csr":
            self._templates.update(CSR_CUDA.conn_templates)
        else:
            raise NotImplementedError
