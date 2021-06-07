import os, sys
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from  tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utillib import utils
import pickle

le = LabelEncoder()

if __name__ == '__main__':
    input_tensor = Input(shape=(64, 64, 3))
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    datas, labels = utils.load_images_with_labels("Data")
    datas = np.array(datas, dtype=np.float)
    labels = le.fit_transform(labels)
    X_train, y_train = datas, labels
    log_dir = os.path.join("TensorboardLogs", "VGG16 " + "Freeze all")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train, validation_split=0.2, epochs=20, verbose=2, callbacks=[tensorboard_callback])
    model.save(os.path.join("Saved Models", "VGG16 " + "Freeze all"))
