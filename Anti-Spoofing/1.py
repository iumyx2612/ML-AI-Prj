from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input, Dense
import sys, os
import datetime
from utillib.utils import load_images_with_labels
import numpy as np


input_tensor = Input(shape=(64, 64, 3))
Resnet = ResNet50(input_tensor=input_tensor, weights="imagenet", include_top=False, pooling='max')
VGG = VGG16(weights="imagenet", include_top=False, pooling='max')
#Inception = InceptionV3(weights="imagenet", include_top=False, pooling='max')

def warmup(base_model: Model, data_dir, img_size=64, epochs=5):
    datas, labels = load_images_with_labels(data_dir, img_size)
    x = base_model.output
    x = Dense(100, activation='relu')(x)
    predictions = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=Resnet.input, outputs=predictions)
    for layer in Resnet.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(datas, labels, epochs=epochs, validation_split=0.2)
    model_timer = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(os.path.join("../Saved Models", "%s " %str(base_model) + model_timer))


