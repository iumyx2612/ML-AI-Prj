from tensorflow.keras.models import Model
from utillib.predict_misc import predict_batch_image


class metrics:
    def __init__(self, name: str = None):
        self.name = name
        self.ground_pos = 0
        self.ground_neg = 0
        self.true_pos = 0
        self.false_pos = 0
        self.true_neg = 0
        self.false_neg = 0

    def run_eval(self, model: Model, test_path="Test", size=64, batch_size=32):
        preds, labels = predict_batch_image(dir=test_path, model=model, size=size, batch_size=batch_size)
        for i in range(len(preds)):
            if preds[i] == labels[i] and labels[i] == "Real":
                self.true_pos += 1
            elif preds[i] == labels[i] and labels[i] == "Spoof":
                self.true_neg += 1
            elif preds[i] != labels[i] and labels[i] == "Spoof":
                self.false_pos += 1
            elif preds[i] != labels[i] and labels[i] == "Real":
                self.false_neg += 1
        self.ground_pos = self.true_pos + self.false_neg
        self.ground_neg = self.true_neg + self.false_pos

    def report(self, name: str, result):
        print("%s evaluation: %s" %(name, result))
        print("----------Done Evaluation----------")

    def cal_tpr(self, model: Model, test_path="Test", size=64, batch_size=32):
        print("----------Starting evaluation----------")
        self.run_eval(model, test_path, size=size, batch_size=batch_size)
        tpr = self.true_pos / self.ground_pos
        self.report("TPR", tpr)

    def cal_fpr(self, model: Model, test_path="Test", size=64, batch_size=32):
        print("----------Starting evaluation----------")
        self.run_eval(model, test_path, size=size, batch_size=batch_size)
        fpr = self.false_pos / self.ground_neg
        self.report("FPR", fpr)

    def cal_acc(self, model: Model, test_path="Test", size=64, batch_size=32):
        print("----------Starting evaluation----------")
        self.run_eval(model, test_path, size=size, batch_size=batch_size)
        acc = (self.true_pos + self.true_neg) / (self.ground_pos + self.ground_neg)
        self.report("Accuracy", acc)
