from nn import MiniVGGNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] accessing CIFAR")
(trainX, trainY), (testX, testY) = cifar10.load_data();


le = LabelBinarizer();
trainY, testY = le.fit_transform(trainY), le.fit_transform(testY);
le.fit_transform(testY);

print(testY);  

print ("[INFO] Compiling model ...");
opt = SGD(lr = 0.01);
model = MiniVGGNet.build(width = 28, height = 28, depth = 1, classes = 10);

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]));