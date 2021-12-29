import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('./kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('./kaggle/input/train.csv')
test_data = pd.read_csv('./kaggle/input/test.csv')
train_data = train_data.dropna(how="any", axis = 0)
X_train = np.array(train_data["x"]).reshape(-1, 1)
y_train = np.array(train_data["y"]).reshape(-1, 1)
X_test = np.array(train_data["x"]).reshape(-1, 1)
y_test = np.array(train_data["y"]).reshape(-1, 1)
y_train = np.array(train_data.iloc[:, -1].values)
# plt.plot(X_train, y_train)
# plt.show()

w = 1.0
def forward(x):
    return w*x
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    l_sum = 0

    for x_val, y_val in zip(X_train, y_train):

        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l

    w_list.append(w)
    mse_list.append(l_sum/3)
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
