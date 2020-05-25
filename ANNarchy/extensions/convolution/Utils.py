# =============================================================================
#
#     Utils.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2019  Julien Vitay <julien.vitay@gmail.com>,
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
# =============================================================================
from ANNarchy.core.Synapse import Synapse

class SharedSynapse(Synapse):
    """
    Shared synapse for report()
    """
    # For reporting
    _instantiated = []

    def __init__(self, psp, operation, name="Shared synapse", description="Weight shared over all synapses of the projection."):
        """
        """
        # Shared synapses are non-plastic.
        Synapse.__init__(self, 
            psp=psp, operation=operation,
            name=name, 
            description=description
        )

        # For reporting
        self._instantiated.append(True)