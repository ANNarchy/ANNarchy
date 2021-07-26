#===============================================================================
#
#     BoldModel.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Oliver Maith <oli_maith@gmx.de>,
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
from ANNarchy.core.Neuron import Neuron

class BoldModel(Neuron):
    """
    Base class to define a bold model to be used in BOLD monitor.
    """
    def __init__(self, parameters, equations, name="bold model", description=""):
        """
        See ANNarchy.extensions.bold.PredefinedModels.py for some example models.

        :param parameters: parameters of the model and their initial value.
        :param equations: equations defining the temporal evolution of variables.
        :param name: optional model name (related to report functionality of ANNarchy)
        :param name: optional model description (related to report functionality of ANNarchy)
        """
        Neuron.__init__(self, parameters=parameters, equations=equations, name=name, description=description)
        self._model_instantiated = False    # activated by BoldMonitor