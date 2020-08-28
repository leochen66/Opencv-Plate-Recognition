import cv2
import numpy as np
from numpy.linalg import norm

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)  
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

# sample = cv2.imread('11-5.jpg', cv2.IMREAD_GRAYSCALE)
# w = abs(sample.shape[1] - 20)
# sample = cv2.copyMakeBorder(sample, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
# sample = cv2.resize(sample, (20, 20), interpolation=cv2.INTER_AREA)
# sample = preprocess_hog([sample])

# model = SVM(C=1, gamma=0.5)
# model.load("svm.dat")
# predict_result = model.predict(sample)
# character = chr(predict_result[0])
# print(character)