from metrics import TPR
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50



if __name__ == '__main__':
    model = load_model("Saved Models/LivenessNet 20210519-005230")
    TPR = TPR()
    TPR.cal_tpr(model)