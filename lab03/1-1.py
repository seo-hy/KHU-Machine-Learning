# pip install scikit-learn
from sklearn import datasets, svm
import random
d = datasets.load_iris()
# print(d.DESCR)
"""
for i in range(0, len(d.data)):
    print(i+1, d.data[i],d.target[i])
"""


s = svm.SVC(gamma=0.1,C=10)
s.fit(d.data, d.target)


new_test = []
for i in range(0, 20):
    new_test .append(d.data[random.randrange(0,len(d.data))])

for e in new_test:
    print(e)


res = s.predict(new_test)
print("새로운 20개 샘플의 부류는", res,"\n")

idx = random.randrange(0,20)
print("new random sample index :",idx)
new_test[idx] = [0.8,1.9,6.5,6.0]

print(new_test[idx])
for e in new_test:
    print(e)
res = s.predict(new_test)
print("변형된 샘플의 부류는", res)


