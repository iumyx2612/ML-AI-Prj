from tensorflow.keras.models import load_model


def inspect(model_path: str):
    model = load_model(model_path)
    model.summary()
    for layer in model.layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print("Layer %s " %layer.name + str(filters.shape))