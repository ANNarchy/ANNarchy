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
    Base class to define a BOLD model to be used in a BOLD monitor.

    A BOLD model is quite similar to a regular rate-coded neuron. It gets a weighted sum of inputs with a specific target (e.g. I) and compute a single output variable (called `BOLD` in the predefined models, but it could be `r` as well).

    The main difference is that a BOLD model should also declare which targets are used and which variable is the output:

    ```python
    bold_model = BoldModel(
        parameters = '''
            tau = 1000.
        ''',
        equations = '''
            I = sum(I)
            # ...
            tau * dBOLD/dt = I - BOLD
        ''',
        inputs = ['I'],
        output = 'BOLD'
    )
    ```
    """
    def __init__(self, parameters, equations, inputs, output, name="Custom BOLD model", description=""):
        """
        See ANNarchy.extensions.bold.PredefinedModels.py for some example models.

        :param parameters: parameters of the model and their initial value.
        :param equations: equations defining the temporal evolution of variables.
        :param inputs: list of input signals (e.g. ['I'] or ['I_f', 'I_r']).
        :param output: output variable of the model (e.g. 'BOLD').
        :param name: optional model name.
        :param description: optional model description.
        """
        self._inputs = inputs
        self._output = output

        Neuron.__init__(self, parameters=parameters, equations=equations, name=name, description=description)
        
        self._model_instantiated = False    # activated by BoldMonitor
