# pip install scikit-learn
from sklearn import datasets, svm, tree
d = datasets.load_iris()
s = svm.SVC(gamma=0.1,C=10)
s.fit(d.data, d.target)

res = s.predict(d.data)
correct = [ i for i in range(len(res)) if res[i]==d.target[i]]
accuracy = len(correct)/len(res)
print("SVM 사용했을 때 정확률 =",accuracy*100,"%")

t = tree.DecisionTreeClassifier(random_state=0)
t.fit(d.data, d.target)
res = t.predict(d.data)
correct = [ i for i in range(len(res)) if res[i]==d.target[i]]
accuracy = len(correct)/len(res)
print("Random Forest 사용했을 때 정확률 =",accuracy*100,"%")



