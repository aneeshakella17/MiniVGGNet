
from nn import MiniVGGNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10

model = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes = 10);

print("[INFO] accessing CIFAR")
(trainX, trainY), (testX, testY) = cifar10.load_data();

le = LabelBinarizer();
trainY, testY = le.fit_transform(trainY), le.fit_transform(testY);
le.fit_transform(testY);

print ("[INFO] Compiling model ...");
opt = SGD(lr = 0.01);

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print ("[INFO] BUILDING MODEL")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=128, epochs=20, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]));