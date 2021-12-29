from sklearn import datasets
import matplotlib.pyplot as plt
import random

digit = datasets.load_digits()

plt.figure(figsize=(5,5))


idx_list = [[] for _ in range(10)]
for i in range(len(digit.target)):
    idx = digit.target[i]
    idx_list[idx].append(i)

for i in range(10):
    print("digit dataset 중에서 "+str(i)+"가 있는 index",end=" ")
    print(idx_list[i])
    ran_idx = random.choice(idx_list[i])
    print(str(i)+"가 들어간 index 중 하나 선택 :"+str(ran_idx))
    plt.imshow(digit.images[ran_idx], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


