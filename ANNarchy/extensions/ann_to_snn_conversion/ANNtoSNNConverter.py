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
from ANNarchy.intern import Messages

import numpy as np

from .InputEncoding import CPN, IB, PSO
from .ReadOut import available_read_outs, IaF, IaF_ReadOut, IaF_TTKS, IaF_Acc

class ANNtoSNNConverter :
    r"""
    Converts a pre-trained Keras model `.keras` into an ANNarchy spiking neural network. 

    The implementation of the present module is inspired by the SNNToolbox (Rueckauer et al. 2017), and is largely based on the work of Diehl et al. (2015). We provide several input encodings, as suggested in the work of Park et al. (2019) and Auge et al. (2021).

    **Constraints on the ANN**

    It is not possible to convert any keras model into an ANNarchy SNN: some constraints ahve to be respected.

    * The only allowed layers in the ANN are:
        * `Dense`
        * `Conv2D`
        * `MaxPooling2D`
        * `AveragePooling2D`
        * as well as non-neural layers such as Dropout, Activation, BatchNorm, etc.

    * The layers must **not** contain any bias, even the conv layers:

    ```python
    x = tf.keras.layers.Dense(128, use_bias=False, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding='same', use_bias=False)(x)
    ```

    * The first layer of the network must be an `Input`:

    ```python
    inputs = tf.keras.Input(shape = (28, 28, 1))
    ```

    * Pooling must explicitly be done by `MaxPooling2D`/`AveragePooling2D`, strides are ignored.
    
    Please be aware that the module is very experimental and the conversion may not work for many different reasons. Feel free to submit issues.

    **Processing Queue**

    The pre-trained ANN model to be converted should be saved in keras (extension `.keras`). The saved model is transformed layer by layer into a feed-forward ANNarchy spiking network. The structure of the network remains the same as in the original ANN, while the weights are normalised. Please note that the current implementation focuses primarily on the correctness of the conversion. Computational performance, especially of the converted CNNs, will be improved in future releases.

    :::callout-note
    
    While the neurons are conceptually spiking neurons, there is one specialty: next to the spike event (stored automatically in ANNarchy), each event will be stored in an additional *mask* array. This *mask* value decays in absence of further spike events exponentially. The decay can be controlled by the *mask_tau* parameter of the population. The projections (either dense or convolution) will use this mask as pre-synaptic input, not the generated list of spike events.
    :::

    **Input Encoding**

    * Poisson ("CPN")

    This encoding uses a Poisson distribution where the pixel values of the image will be used as probability for each individual neuron.
    
    * Intrinsically Bursting ("IB")

    This encoding is based on the Izhikevich (2003) model that comprises two ODEs:

    $$
    \begin{cases}
    \frac{dv}{dt} = 0.04 \cdot v^2 + 5.0 \cdot v + 140.0 - u + I \\
    \\
    \frac{du}{dt} = a \cdot (b \cdot v - u) \\
    \end{cases}
    $$

    The parameters for $a$ - $d$ are selected accordingly to Izhikevich (2003). The provided input images will be set as $I$.


    * Phase Shift Oscillation ("PSO")

    Based on the description by Park et al. (2019), the spiking threshold $v_\text{th}$ is modulated by a oscillation function $\Pi$, whereas the membrane potential follows simply the input current. 

    $$
    \begin{cases}
    \Pi(t) = 2^{-(1+ \text{mod}(t,k))}\\
    \\
    v_\text{th}(t) = \Pi(t) \, v_\text{th}(t)\\
    \end{cases}
    $$

    * User-defined input encodings

    In addition to the pre-defined models, one can opt for individual models using the `Neuron` class of ANNarchy. Please note that a `mask` variable need to be defined, which is fed into the subsequent projections.

    **Read-out Methods**

    In a classification task, the neuron with the highest activity corresponds corresponds to the decision to which class the presented input belongs. However, the highest activity can be determined in different ways. We support currently three methods, defined by the `read_out` parameter of the constructor:

    * Maximum Spike Count

    `read_out = 'spike_count'` : the number of spikes emitted by each neuron is recorded and the index of the neuron(s) with the maximum number is returned.

    * Time to Number of Spikes

    `read_out = 'time_to_first_spike'` or `read_out = 'time_to_k_spikes'`: when the first or first $k$ spikes are emitted by a single neuron, the simulation is stopped and the neuron rank(s) is returned. For the second mode, an additional $k$ argument need to be also provided.

    * Membrane potential

    `read_out = 'membrane_potential'`:  pre-synaptic events are accumulated in the membrane potential of each output neuron. The index of the neuron(s) with the highest membrane potential is returned.

    > Izhikevich (2003) Simple Model of Spiking Neurons. IEEE transactions on neural networks 14(6). doi: 10.1109/TNN.2003.820440

    > Diehl PU, Neil D, Binas J,et al. (2015) Fast-classifying, high-accuracy spiking deep networks through weight and threshold balancing, 2015 International Joint Conference on Neural Networks (IJCNN), 1-8, doi: 10.1109/IJCNN.2015.7280696.

    > Rueckauer B, Lungu I, Hu Y, et al. (2017) Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification., Front. Neurosci., 2017, 11. doi: 10.3389/fnins.2017.00682

    > Park S, Kim S, Choe H, et al. (2019) Fast and Efficient Information Transmission with Burst Spikes in Deep Spiking Neural Networks. 

    > Auge D, Hille J, Mueller E et al. (2021) A Survey of Encoding Techniques for Signal Processing in Spiking Neural Networks. Neural Processing Letters. 2021; 53:4963-4710. doi:10.1007/s11063-021-10562-2

    :param input_encoding: a string representing which input encoding should be used: 'CPN', 'IB' or 'PSO'.
    :param hidden_neuron:  neuron model used in the hidden layers. Either the default integrate-and-fire ('IaF') or an ANNarchy Neuron object.
    :param read_out: a string which of the following read-out method should be used: `spike_count`, `time_to_first_spike`, `membrane_potential`.
    """

    def __init__(self, 
                 input_encoding:str='CPN', 
                 hidden_neuron:str='IaF', 
                 read_out:str='spike_count', 
                 **kwargs):

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
                Messages._error("When read_out is set to 'time_to_k_spikes', the k parameter need to be provided.")
            self._k_param = kwargs['k']

        # TODO: sanity check on key-value args
        for key, value in kwargs.items():
            if key == "max_f":
                self._max_f = value

        self._snn_network = None

    def load_keras_model(self, 
            filename:str, 
            directory:str="annarchy", 
            scale_factor:float=None, 
            show_info:bool=True, 
        ) -> "Network":
        """
        Loads the pre-trained model provided as a .keras file.

        In tf.keras, the weights can be saved using:

        ```python
        model.save("model.keras")
        ```

        :param filename: path to the `.keras` file.
        :param directory: sub-directory where the generated code should be stored (default: "annarchy")
        :param scale_factor: allows a fine-grained control of the weight scale factor. By default (None), with each layer-depth the factor increases by one. If a scalar value is provided the same value is used for each layer. Otherwise a list can be provided to assign the scale factors individually.
        :param show_info: whether the network structure should be printed on console (default: True)
        :returns: An `ANNarchy.Network` instance.
        """
        description = ""

        # Load keras model
        import tensorflow as tf
        model = tf.keras.models.load_model(filename)

        # Create spiking network
        self._snn_network = Network(everything = False)

        # Input Population
        if not isinstance(model.layers[0], tf.keras.layers.InputLayer):
            Messages._error("The first layer of the network must be an Input layer.")
        
        input_name = model.layers[0].name
        input_shape = get_shape(model.layers[0].output)[1:]

        input_pop = Population(
            name = input_name, 
            geometry=input_shape, 
            neuron=self._input_model
        )
        self._snn_network.add(input_pop)

        description += f"* Input layer: {input_name}, {input_shape}\n"

        # Iterate over layers
        pops = [input_pop]
        projs = []
        weights = []
        
        for idx, layer in enumerate(model.layers):

            # Get the name
            name = layer.name

            # Is the layer legit in ANNarchy?
            pop = None

            # The only supported layers are dense, conv* and pool*
            if isinstance(layer, tf.keras.layers.Dense):
                
                # Get incoming weights
                W = layer.get_weights()[0].T
                size = W.shape[0]

                # Neuron type
                neuron_type = self._hidden_neuron_model
                stop_condition = None

                # Readout layer can have different neurons
                readout = False
                if idx == len(model.layers) - 1:
                    readout = True
                    if self._read_out == "membrane_potential":
                        neuron_type = IaF_Acc 
                        stop_condition = None
                    elif self._read_out == "time_to_first_spike":
                        neuron_type = IaF
                        stop_condition="spiked: any"
                    elif self._read_out == "time_to_k_spikes":
                        neuron_type = IaF_TTKS 
                        stop_condition="sc >= k: any"
                
                # Create the population
                pop = Population(
                    geometry = size, 
                    neuron=neuron_type, 
                    name=name,
                    stop_condition=stop_condition
                )

                # Add population to the network
                self._snn_network.add(pop)
                pops.append(pop)

                if not readout:
                    #pop.vt = pop.vt - (0.05 * len(pops))
                    pop.vt = pop.vt - (0.05 * idx)

                description += f"* Dense layer: {name}, {size} \n"

                # Create projection
                proj = Projection(
                    pre = pops[-2], post = pop, 
                    target = "exc", name=f"dense_proj_{len(pops)}"
                )

                proj.connect_from_matrix(W, storage_format="dense")

                # Add to the network
                self._snn_network.add(proj)
                projs.append(proj)
                weights.append(W)

                description += f"    weights: {W.shape}\n"
                description += f"    mean {np.mean(W)}, std {np.std(W)}\n"
                description += f"    min {np.min(W)}, max {np.max(W)}\n"
                

            elif isinstance(layer, tf.keras.layers.Conv2D):
                
                geometry = get_shape(layer.output)[1:]

                pop = Population(
                    geometry = geometry, 
                    neuron=self._hidden_neuron_model, 
                    name=name ) 

                self._snn_network.add(pop)
                pops.append(pop)
                
                #pop.vt = pop.vt - (0.05 * len(pops))
                pop.vt = pop.vt - (0.05 * idx)

                description += f"* Conv2D layer: {name}, {geometry} \n"

                W = layer.get_weights()[0]
                W = np.moveaxis(W, -1, 0)

                proj = Convolution(
                    pre = pops[-2], post = pop, target='exc', 
                    psp="pre.mask * w", 
                    name=f'conv_proj_{len(pops)}')
                proj.connect_filters(weights=W)

                self._snn_network.add(proj)
                projs.append(proj)
                weights.append(W)

            elif isinstance(layer, 
                            (tf.keras.layers.MaxPooling2D,
                             tf.keras.layers.AveragePooling2D)):
                
                geometry = get_shape(layer.output)[1:]
                pool_size = layer.get_config()['pool_size'] + (1,)
                operation = 'max' if isinstance(layer, tf.keras.layers.MaxPooling2D) else 'mean'

                pop = Population(
                    geometry = geometry, 
                    neuron=self._hidden_neuron_model, 
                    name=name ) 

                self._snn_network.add(pop)
                pops.append(pop)
                
                #pop.vt = pop.vt - (0.05 * len(pops))
                pop.vt = pop.vt - (0.05 * idx)

                description += f"* MaxPooling2D layer: {name}, {geometry} \n"
                
                proj = Pooling(
                    pre = pops[-2], post = pop, target='exc', 
                    operation=operation, psp="pre.mask", 
                    name=f'pool_proj_{len(pops)}')
                
                proj.connect_pooling(extent=pool_size)
                self._snn_network.add(proj)
                projs.append(proj)
                weights.append([])

            # Compatible layer has not been found
            if pop is None:
                description += f"* {type(layer).__name__} skipped.\n"

        # Record the last layer to determine prediction
        self._monitor = Monitor(pop, ['v', 'spike'])
        self._snn_network.add(self._monitor)

        # Compile the configured network
        self._snn_network.compile(directory=directory, silent=True)

        if show_info:
            print(description)

        # Normalize the weights
        factors = self._normalize_weights(weights, scale_factor)
        for i in range(len(weights)):
            if len(weights[i]) > 0:
                if isinstance(projs[i], (Convolution,)):
                    self._snn_network.get(projs[i]).weights = weights[i] * float(factors[i])
                else:
                    self._snn_network.get(projs[i]).weights = weights[i] * float(factors[i])

        return self._snn_network


    def get_annarchy_network(self) -> "Network":
        """
        Returns the ANNarchy.Network instance.
        """
        return self._snn_network

    def predict(self, 
                samples, 
                duration_per_sample=1000, 
                multiple=False,
                ) -> list[int]:
        """
        Performs the prediction for a given input array.

        :param samples: set of inputs to present to the network. The function expects a 2-dimensional array (num_samples, input_size).
        :param duration_per_sample: the number of simulation steps for one input sample (default: 1000, 1 second biological time)
        :param multiple: if several output neurons reach the criteria, return the full list instead of randomly chosing one.
        :returns: A list of predicted class indices for each sample.
        """

        predictions = []

        # Get the top-level layer
        first_layer = self._snn_network.get_populations()[0].name
        last_layer = self._snn_network.get_population(self._snn_network.get_populations()[-1].name)

        nb_classes = last_layer.size

        # Use the progress bar
        try:
            import tqdm
        except Exception as e:
            iterator = range(samples.shape[0])
        else:
            iterator = tqdm.tqdm(range(samples.shape[0]))

        # Iterate over all samples
        for i in iterator:

            # Reset state variables
            self._snn_network.reset(populations=True, monitors=True, projections=False)

            # Set input
            self._snn_network.get_population(first_layer).rates =  samples[i,:] * self._max_f

            # The read-out is performed differently based on the mode selected by the user
            if self._read_out in ["time_to_first_spike", "time_to_k_spikes"]:
                # Simulate until the condition is met
                self._snn_network.simulate_until(duration_per_sample, population=last_layer)

                # Read-out accumulated inputs
                data = self._snn_network.get(self._monitor).get('spike')
                act_pred = np.zeros(nb_classes)
                for neur_rank, spike_times in data.items():
                    act_pred[neur_rank] = len(spike_times)

                # Gather all neurons which fulfilled the condition
                prediction = np.argwhere(act_pred == np.amax(act_pred)).flatten()

            elif self._read_out == "membrane_potential":
                # Simulate 1s and record spikes in output layer
                self._snn_network.simulate(duration_per_sample)

                # Read-out accumulated inputs
                data = self._snn_network.get(self._monitor).get('v')

                # The neuron with the highest accumulated membrane potential is the selected candidate
                prediction = np.argwhere(data[-1,:] == np.amax(data[-1,:])).flatten()

            elif self._read_out == "spike_count":
                # Simulate 1s and record spikes in output layer
                self._snn_network.simulate(duration_per_sample)

                # Retrieve the recorded spike events
                data = self._snn_network.get(self._monitor).get('spike')

                # The predicted label is the neuron index with the highest number of spikes.
                # Therefore, we count the number of spikes each output neuron emitted.
                act_pred = np.zeros(nb_classes)
                for neur_rank, spike_times in data.items():
                    act_pred[neur_rank] = len(spike_times)

                # Gather all neurons which achieved highest number of spikes
                prediction = np.argwhere(act_pred == np.amax(act_pred)).flatten()

            else:
                raise NotImplementedError
            
            # Treat ex-aequo
            if multiple:
                predictions.append(list(prediction))
            else:
                predictions.append(int(np.random.choice(prediction, 1)))

        return predictions


    def _normalize_weights(self, weight_matrices, scale_factor=None):
        """
        Weight normalization based on the "model based normalization" from Diehl et al. (2015)
        """

        # Scale factor increases for each layer
        if scale_factor is None:
            scale_factor = np.arange(2, len(weight_matrices)+2)
        elif isinstance(scale_factor, (float, int)):
            scale_factor = [scale_factor] * len(weight_matrices)
        elif isinstance(scale_factor, (list, np.ndarray)):
            if len(scale_factor) != len(weight_matrices):
                Messages._error("The length of the scale_factor list must be equal to the number of projections.")
            else:
                pass # nothing to do
        else:
            raise ValueError("Invalid argument for scale_factor", type(scale_factor))

        # Iterate over all weight matrices
        factors = []
        for level, w_matrix in enumerate(weight_matrices):

            factor = 1.0

            if len(w_matrix)> 0: # Empty weight matrix is for max-pooling

                # Reshape the matrix with post-neurons first
                w = w_matrix.reshape((w_matrix.shape[0], -1))

                # Maximum input current over all post neurons
                max_val = np.sum(w * (w > 0), axis=1).max()

                # Normalize the incoming weights for each neuron, based on the maximum input for the complete connection 
                # and multiply it with the depth of the connection to boost the input current
                factor = scale_factor[level]  / max_val

            factors.append(factor)

        return factors

def get_shape(tensor) -> tuple:
    """
    Returns the shape of the tensorflow tensor as a tuple.

    tf < 2.14 returned a TensorShape object, but now it is a tuple.
    
    """
    if isinstance(tensor.shape, (tuple,)):
        return tensor.shape
    else:
        try:
            return tuple(tensor.shape.as_list())
        except:
            Messages._error("ANN_to_SNN: unable to estimate the layer's size.")

