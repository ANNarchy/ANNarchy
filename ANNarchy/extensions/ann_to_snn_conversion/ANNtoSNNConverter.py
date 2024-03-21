"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core import Global
from ANNarchy.core.Network import Network
from ANNarchy.core.Population import Population
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Monitor import Monitor
from ANNarchy.core.Random import Uniform
from ANNarchy.extensions.convolution import Convolution, Pooling


import matplotlib.pylab as plt
import numpy as np
import h5py
import json
from copy import copy

from .InputEncoding import *
from .ReadOut import *

class ANNtoSNNConverter :
    """
    Implements a conversion of a pre-trained fully-connected Keras model into a spiking model. The procedure is based on the implementation of:
     
    > Diehl et al. (2015) "Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing" Proceedings of IJCNN. doi: 10.1109/IJCNN.2015.7280696

    """

    def __init__(self, input_encoding='poisson', hidden_neuron='IaF', read_out='spike_count', **kwargs):
        """
        :param input_encoding: a string representing which input encoding should be used: custom poisson, PSO, IB or CH (for more details see InputEncoding).
        :param hidden_neuron:  neuron model used in the hidden layers. Either the default integrate-and-fire ('IaF') or an ANNarchy Neuron object.
        :param read_out: a string which of the following read-out method should be used: spike_count, time_to_first_spike, membrane_potential (for more details see the manual).
        """

        # Neuron model
        if isinstance(hidden_neuron, str):
            if hidden_neuron == "IaF":
                self._hidden_neuron_model = IaF
            else:
                raise ValueError("Invalid model name for hidden neurons.")
        else:
            self._hidden_neuron_model = hidden_neuron

        # Input encoding
        self._input_encoding = input_encoding
        if input_encoding == "poisson" or input_encoding == "CPN":
            self._input_model = CPN
        elif input_encoding == 'PSO':
            self._input_model = PSO
        elif input_encoding=='IB':
            self._input_model = IB
        else:
            raise ValueError("Unknown input encoding:", input_encoding)

        # Readout
        if read_out in available_read_outs:
            self._read_out = read_out
        else:
            raise ValueError("Unknown value for read-out:", read_out)

        # Maximum frequency
        self._max_f = 100      # scale factor used for poisson encoding

        if self._read_out == "time_to_k_spikes":
            if 'k' not in kwargs.keys():
                Global._error("When read_out is set to 'time_to_k_spikes', the k parameter need to be provided.")
            self._k_param = kwargs['k']

        # TODO: sanity check on key-value args
        for key, value in kwargs.items():
            if key == "max_f":
                self._max_f = value

        self._snn_network = None

    def init_from_keras_model(self, 
            filename, 
            directory="annarchy", 
            scale_factor=None, 
            show_info=True, 
        ):
        """
        Loads the pre-trained model provided as a .h5 file.

        In tf.keras, the weights can be saved using:

        ```python
        model.save("model.h5")
        ```

        :param filename: stored model as a .h5 file.
        :param directory: sub-directory where the generated code should be stored (default: "annarchy")
        :param scale_factor: allows a fine-grained control of the weight scale factor. By default (None), with each layer-depth the factor increases by one. If a scalar value is provided the same value is used for each layer. Otherwise a list can be provided to assign the scale factors individually.
        :param show_info: whether the network structure should be printed on console (default: True)

        :returns net: An `ANNarchy.Network` instance.
        """

        # Filename
        if not filename.endswith(".h5"):
            Global._error("ANNtoSNNConverter: the keras model must be provided as a .h5 file.")
        self._filename = filename

        # Extract weight matrices
        weight_matrices, layer_order, layer_operation, input_dim = self._extract_weight_matrices(filename)

        # Create spiking network
        self._snn_network = Network(everything = False)
        input_pop = Population(name = layer_order[0], geometry=input_dim, neuron=self._input_model)
        self._snn_network.add(input_pop)

        # Hidden neuron
        if isinstance(self._hidden_neuron_model, list):
            hidden_type = "user-specified"
        else:
            hidden_type = self._hidden_neuron_model.name
            self._hidden_neuron_model = [self._hidden_neuron_model for _ in range(len(layer_order)-2)]

        description = f"""Parameters
----------------------
* input encoding: {self._input_encoding}
* hidden neuron: {hidden_type}
* read-out method: {self._read_out}

Layers
----------------------
"""

        # Create populations
        for layer in range(len(layer_order)):
            if 'conv' in layer_order[layer]:

                geometry = input_dim + (np.shape(weight_matrices[layer])[0],)
                conv_pop = Population(geometry = geometry, neuron=self._hidden_neuron_model[layer-1], name=layer_order[layer] ) # create convolution population
                conv_pop.vt = conv_pop.vt - (0.05*layer) # reduce the threshold for deeper layers
                self._snn_network.add(conv_pop)

                description += f"* name={layer_order[layer]}, convolutional layer, {geometry=}\n"

            elif 'pool' in layer_order[layer]:

                input_dim = (int(input_dim[0]/ 2), int(input_dim[1]/ 2))
                l_weights = weight_matrices[layer-1] # get the weights of the previous layer (should be a conv-layer, or not?)
                dim_0 = np.shape(l_weights)[0]

                geometry = input_dim # add it to the geometry
                geometry = geometry + (dim_0,)
                pool_pop = Population(geometry = geometry, neuron=self._hidden_neuron_model[layer-1], name=layer_order[layer])
                pool_pop.vt = pool_pop.vt - (0.05*layer) # reduce the threshold for deeper layers
                self._snn_network.add(pool_pop)

                description += f"* name={layer_order[layer]}, pooling layer, {geometry=}\n"

            elif 'dense' in layer_order[layer]:

                geometry = np.shape(weight_matrices[layer])[0]

                # read-out layer
                if layer == len(layer_order)-1:
                    if self._read_out == "membrane_potential":
                        # HD (24th April 2023): instead of reading out spike events, we use the accumulated inputs
                        #                       as decision parameter
                        dense_pop = Population(geometry = geometry, neuron=IaF_Acc, name=layer_order[layer])

                    elif self._read_out == "time_to_first_spike" and layer == len(layer_order)-1:
                        # HD (24th April 2023): instead of reading out spike events, we simulate untile the first
                        #                       neuron in the read-out layer spikes.
                        dense_pop = Population(geometry = geometry, neuron=IaF, name=layer_order[layer], stop_condition="spiked: any")

                    elif self._read_out == "time_to_k_spikes" and layer == len(layer_order)-1:
                        # HD (3rd May 2023): instead of reading out spike events, we simulate untile the first
                        #                    neuron emitted k spikes in the read-out layer.
                        dense_pop = Population(geometry = geometry, neuron=IaF_TTKS, name=layer_order[layer], stop_condition="sc >= k: any")
                        dense_pop.k = self._k_param

                    else:
                        dense_pop = Population(geometry = geometry, neuron=IaF_ReadOut, name=layer_order[layer])
                        # ARK:  scaling the threshold as number of layers increases divide
                        #       the value 1/half of the number of the network
                        dense_pop.vt = dense_pop.vt - (0.05*layer)
                        # HD (20th Feb. 2023): we want to generate this firing vector for a
                        #                      single time step
                        dense_pop.compute_firing_rate(Global.dt())

                # hidden layer neurons
                else:
                    dense_pop = Population(geometry = geometry, neuron=self._hidden_neuron_model[layer-1], name=layer_order[layer])
                    # ARK:  scaling the threshold as number of layers increases divide
                    #       the value 1/half of the number of the network
                    dense_pop.vt = dense_pop.vt - (0.05*layer)
                    # HD (20th Feb. 2023): we want to generate this firing vector for a
                    #                      single time step
                    dense_pop.compute_firing_rate(Global.dt())

                # Add created layer to the network
                self._snn_network.add(dense_pop)

                # Description
                description += f"* name={layer_order[layer]}, dense layer, {geometry=}\n"


        # Create Projections
        description += '\nProjections\n----------------------\n'

        for p in range(1,len(layer_order)):
            if 'conv' in layer_order[p]:

                post_pop = self._snn_network.get_population(layer_order[p])
                pre_pop = self._snn_network.get_population(layer_order[p-1])

                weight_m = np.squeeze(weight_matrices[p])

                conv_proj = Convolution(pre = pre_pop, post=post_pop, target='exc', psp="pre.mask * w", name='conv_proj_%i'%p)
                conv_proj.connect_filters(weights=weight_m)
                self._snn_network.add(conv_proj)

                description += f"* {layer_order[p-1]} {pre_pop.geometry} -> {layer_order[p]} {post_pop.geometry}\n"
                description += f"    weight matrix {np.shape(weight_m)}\n"


            elif 'pool' in layer_order[p]:

                post_pop = self._snn_network.get_population(layer_order[p])
                pre_pop = self._snn_network.get_population(layer_order[p-1])

                pool_proj = Pooling(pre = pre_pop, post=post_pop, target='exc', operation=layer_operation[p], psp="pre.mask", name='pool_proj_%i'%p)
                pool_proj.connect_pooling(extent=(2,2,1))
                self._snn_network.add(pool_proj)

                description += f"* {layer_order[p-1]} {pre_pop.geometry} -> {layer_order[p]} {post_pop.geometry}\n"
                description += f"    pooling operation {layer_operation[p]}\n"

            elif 'dense' in layer_order[p]:

                post_pop = self._snn_network.get_population(layer_order[p])
                pre_pop = self._snn_network.get_population(layer_order[p-1])

                dense_proj = Projection(
                    pre = pre_pop, post = post_pop, 
                    target = "exc", name='dense_proj_%i'%p
                )

                if pre_pop.neuron_type.type=="rate":
                    dense_proj.connect_all_to_all(weights=Uniform(0,1), storage_format="dense")
                else:
                    dense_proj.connect_all_to_all(weights=Uniform(0,1), storage_format="dense", storage_order="pre_to_post")
                    dense_proj._parallel_pattern = "outer_loop"

                self._snn_network.add(dense_proj)

                description += f"* {layer_order[p-1]} {pre_pop.geometry} -> {layer_order[p]} {post_pop.geometry}\n"
                description += f"    weight matrix size {np.shape(weight_matrices[p])}\n"
                description += f"    mean {np.mean(weight_matrices[p])}, std {np.std(weight_matrices[p])}\n"
                description += f"    min {np.min(weight_matrices[p])}, max {np.max(weight_matrices[p])}\n"

        # Compile the configured network
        self._snn_network.compile(directory=directory)

        # Weight normalization
        self._apply_weight_normalization(scale_factor, show_info)

        if show_info:
            print(description)

        return self._snn_network

    def _apply_weight_normalization(self, scale_factor=None, show_info=True):
        """
        Apply the weight normalization as described in Diehl et. al (2015). Note that the function is automatically called by default.
        """
        
        # 1st step: load original ANN
        weight_matrices, layer_order, layer_operation, input_dim = self._extract_weight_matrices(self._filename)

        # 2nd step: normalize weights
        norm_weight_matrices = self._normalize_weights(weight_matrices, scale_factor=scale_factor)

        ## go again over all dense projections to load the weight matrices ##
        for proj in self._snn_network.get_projections():
            
            if 'dense' in proj.name: # find the dense projection
                proj_name = proj.name.split('_')
                proj_idx = int(proj_name[-1]) # get the index of the dense layer in relation to all other layers

                ## use the not normed weights to the classification layer
                proj.w = norm_weight_matrices[proj_idx]


    def get_annarchy_network(self):
        """
        Returns the ANNarchy.Network instance.
        """
        return self._snn_network

    def predict(self, 
                samples, 
                duration_per_sample=1000, 
                measure_time=False,
                ):
        """
        Performs the prediction for a given input series.

        Parameters:

        :param samples: set of inputs to present to the network. The function expects a 2-dimensional array (num_samples, input_size).
        :param duration_per_sample: the number of simulation steps for one input sample (default: 1000, 1 second biological time)
        :param measure_time: print out the computation time spent for one input sample (default: False)

        :returns predictions: A list of predicted class indices. If multiple neurons fulfill the condition, all candidate indices are returned.
        """

        predictions = [[] for _ in range(len(samples))]

        # get the top-level layer
        first_layer = self._snn_network.get_populations()[0].name
        last_layer = self._snn_network.get_population(self._snn_network.get_populations()[-1].name)

        # record the last layer to determine prediction
        if self._read_out == "membrane_potential":
            m_read_out_layer = Monitor(last_layer, ['v'])
        else:
            m_read_out_layer = Monitor(last_layer, ['spike'])
        self._snn_network.add(m_read_out_layer)

        class_pop_size = last_layer.size

        # Needed when multiple classes achieve the same ranking
        rng = np.random.default_rng()

        # Iterate over all samples
        for i in range(samples.shape[0]):
            # Progress bar
            if i % 100 == 0:
                print(f"{i}/{samples.shape[0]}", end="\r")

            # Reset state variables
            self._snn_network.reset(populations=True, monitors=True, projections=False)

            # set input
            self._snn_network.get_population(first_layer).rates =  samples[i,:]*self._max_f

            # The read-out is performed differently based on the mode selected by the user
            if self._read_out in ["time_to_first_spike", "time_to_k_spikes"]:
                self._snn_network.simulate_until(duration_per_sample, population=last_layer, measure_time=measure_time)

                # read-out accumulated inputs
                spk_class = self._snn_network.get(m_read_out_layer).get('spike')
                act_pred = np.zeros(class_pop_size)
                for neur_rank, spike_times in spk_class.items():
                    act_pred[neur_rank] = len(spike_times)

                # gather all neurons which fulfilled the condition
                tmp_list = np.argwhere(act_pred == np.amax(act_pred))
                for tmp in tmp_list:
                    predictions[i].append(tmp[0])

            elif self._read_out == "membrane_potential":
                # simulate 1s and record spikes in output layer
                self._snn_network.simulate(duration_per_sample, measure_time=measure_time)

                # read-out accumulated inputs
                spk_class = self._snn_network.get(m_read_out_layer).get('v')

                # the neuron with the highest accumulated membrane potential is the selected candidate
                # (HD: 23th May 2023: I'm not sure if it could happen that two neurons have the same mp)
                tmp_list = np.argwhere(spk_class[-1,:] == np.amax(spk_class[-1,:]))
                for tmp in tmp_list:
                    predictions[i].append(tmp[0])

            elif self._read_out == "spike_count":
                # simulate 1s and record spikes in output layer
                self._snn_network.simulate(duration_per_sample, measure_time=measure_time)

                # retrieve the recorded spike events
                spk_class = self._snn_network.get(m_read_out_layer).get('spike')

                # The predicted label is the neuron index with the highest number of spikes.
                # Therefore, we count the number of spikes each output neuron emitted.
                act_pred = np.zeros(class_pop_size)
                for neur_rank, spike_times in spk_class.items():
                    act_pred[neur_rank] = len(spike_times)

                # gather all neurons which achieved highest number of spikes
                tmp_list = np.argwhere(act_pred == np.amax(act_pred))
                for tmp in tmp_list:
                    predictions[i].append(tmp[0])

            else:
                raise NotImplementedError

        return predictions

    def _set_input(self, sample, encoding):
        """
        We will support different input encodings in the future.
        """
        if encoding == "poisson":
            self._input_pop.rates = sample*self._max_f
        else:
            self._input_pop.rates = sample*self._max_f

    def _extract_weight_matrices(self, filename):
        """
        Read the .h5 file and extract layer sizes as well as the
        pre-trained weights.
        """
        f=h5py.File(filename,'r')

        if not 'model_weights' in f.keys():
            Global._error("Could not find weight matrices in the .h5 file.")

        ## get the configuration of the Keras model
        model_config = f.attrs.get("model_config")
        try:
            # h5py < 3.0 get() returns 'bytes' sequence
            model_config = model_config.decode("utf-8")
        except AttributeError:
            # In h5py > 3.0 the return of get() is already decoded
            pass

        model_config = json.loads(model_config)

        ## get the list with all layer names
        model_layers = (model_config['config']['layers'])
        model_weights = (f['model_weights'])

        Global._debug("ANNtoSNNConverter: detected", len(model_layers), "layers.")

        weight_matrices=[]   # array to save the weight matrices
        layer_order = []     # additional array to save the order of the layers to know it later
        layer_operation = [] # additional information for each layer

        for layer in model_layers:
            layer_name = layer['config']['name']
            layer_class = layer['class_name']

            if 'conv2d' in layer_name:
                layer_w = model_weights[layer_name][layer_name]['kernel:0']
                ## if it is a convolutional layer, reshape it to fitt to annarchy
                dim_h, dim_w, dim_pre, dim_post = np.shape(layer_w)
                new_w = np.zeros((dim_post, dim_h, dim_w, dim_pre))
                for i in range(dim_post):
                    new_w[i,:,:] = layer_w[:,:,:,i]
                weight_matrices.append(new_w)
                layer_order.append(layer_name)
                layer_operation.append("") # add an empty string to pad the array

            elif 'dense' in layer_name:
                layer_w = model_weights[layer_name][layer_name]['kernel:0']
                weight_matrices.append(np.transpose(layer_w))
                layer_order.append(layer_name)
                layer_operation.append("") # add an empty string to pad the array

            elif 'pool' in layer_name:
                layer_order.append(layer_name)
                weight_matrices.append([]) # add an empty weight matrix to pad the array
                if "AveragePooling" in layer_class:
                    layer_operation.append("mean")
                elif "MaxPooling" in layer_class:
                    layer_operation.append("max")
                else:
                    Global._warning("The pooling class:", layer_class, "is not supported yet. Falling back to max-pooling.")
                    layer_operation.append("max")

            elif 'input' in layer_name:
                layer_order.append(layer_name)
                input_dim = layer['config']['batch_input_shape']
                if len(input_dim) >2 : #probably a conv. if >2
                    input_dim = tuple(input_dim[1:3])
                else:           # probably a MLP
                    input_dim = input_dim[1]
                weight_matrices.append([]) # add an empty weight matrix to pad the array
                layer_operation.append("")

        return weight_matrices, layer_order, layer_operation, input_dim

    def _normalize_weights(self, weight_matrices, scale_factor=None):
        """
        Weight normalization based on the "model based normalization" from Diehl et al. (2015)
        """
        norm_wlist=[]

        ## Argument checking
        if scale_factor is None:
            scale_factor = np.arange(1, len(weight_matrices)+1)
        elif isinstance(scale_factor, (float, int)):
            scale_factor = [scale_factor] * len(weight_matrices)
        elif isinstance(scale_factor, (list, np.array)):
            if len(scale_factor) != len(weight_matrices):
                Global._error("The length of the scale_factor list must be equal the number of projections.")
            else:
                pass # nothing to do
        else:
            raise ValueError("Invalid argument for scale_factor", type(scale_factor))

        ## iterate over all weight matrices
        for level in range(len(weight_matrices)):
            max_pos_input = 0
            w_matrix = copy(weight_matrices[level])
            if len(w_matrix)> 0: # Empty weight matrix ?
                ## each row correspnds to one post-synaptic neuron
                for row in range (w_matrix.shape[0]):
                    w_matrix_flat=w_matrix[row].flatten()
                    idx=np.where(w_matrix_flat>0)
                    input_sum=np.sum(w_matrix_flat[idx])
                    ## save the maximum input current over all post neurons in this connection
                    max_pos_input = max(max_pos_input, input_sum)

                for row in range (w_matrix.shape[0]):
                    ## normalize the incoming weights for each neuron, based on the maximum input
                    ## for the complete connection
                    ## and multiply it with the deepth of the connection to boost the input current
                    w_matrix[row]=scale_factor[level]* w_matrix[row]/max_pos_input

            norm_wlist.append(w_matrix)

        return norm_wlist