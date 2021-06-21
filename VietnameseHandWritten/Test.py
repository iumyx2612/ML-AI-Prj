import tensorflow as tf
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("SavedModel/MTDMTnet 20210428-003826")
model.summary()
for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)

'''filters, biases = model.layers[0].get_weights()
n_filters = 9
ix = 1
for i in range(filters.shape[-1]):
    filter = filters[:, :, :, i]
    #fig = plt.figure(figsize=(4, 4))
    for j in range(filters.shape[2]):
        ax = plt.subplot(4, 4, ix)
        plt.imshow(filter[:, :, j], cmap='gray')
        ix += 1
plt.show()'''


