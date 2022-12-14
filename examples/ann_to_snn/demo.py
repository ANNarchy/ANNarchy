#
#   Simple demonstration of the ANNtoSNNConverter
#
#   Authors:    Abdul Rehaman Kampli, Helge Uelo Dinkelbach and René Larisch
#
from ANNarchy.extensions.ann_to_snn_conversion import ANNtoSNNConverter
from mnist.loader import MNIST
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load and prepare MNIST input
mndata = MNIST("./mnist_testset")
images, labels = mndata.load_testing()

ab_img=np.array(images)
ab_labl=np.array(labels)

xt=ab_img.reshape(10000, 784) / 255

# Initialize ANNarchy SNN network with pre-trained weights
# Implemented Encoding Technichs IB,CH and PSO

snn_converter = ANNtoSNNConverter(input_encoding='PSO')
snn_converter.init_from_keras_model("model_dense.h5")

# test the first 50 samples
predictions = snn_converter.predict(xt[:50,:], duration_per_sample=100)

#Classification Report and Accuracy of the network
print(classification_report(ab_labl[:50], predictions))
print(accuracy_score(ab_labl[:50], predictions))

