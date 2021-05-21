from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from training import LivenessNet

predefined_net = [
    "LivenessNet",
    "VGG16",
    "VGG19",
    "Inception_v3"
]