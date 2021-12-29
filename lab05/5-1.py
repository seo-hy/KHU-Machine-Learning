# pip3 install torch torchvision torchaudio

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, validation_curve
import numpy as np
from sklearn.datasets import fetch_openml
import time

mnist = fetch_openml("mnist_784")
mnist.data = mnist.data/255.0
x_train = mnist.data[:60000]; x_test = mnist.data[60000:]
y_train = np.int16(mnist.target[:60000]); y_test = np.int16(mnist.target[60000:])

start = time.time()
mlp = MLPClassifier(learning_rate_init=0.001, batch_size=32, max_iter=3, solver="sgd")
prange = range(50, 101, 50)

train_score,test_score = validation_curve(mlp, x_train, y_train, param_name="hidden_layer_sizes", param_range=prange, cv=10, scoring="accuracy",n_jobs=4)
end = time.time()
print("하이퍼 매개변수 최적화에 걸린 시간은", end-start, "초입니다.")