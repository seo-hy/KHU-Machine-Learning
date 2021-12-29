from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np

digit=datasets.load_digits()
s=svm.SVC(gamma=0.001)
for i in range(5,10):
    print(str(i)+"겹 교차검증")
    accuracies = cross_val_score(s, digit.data, digit.target, cv=i)
    print(accuracies)
    print("정확률(평균)=%0.3f, 표준편차=%0.3f"%(accuracies.mean()*100, accuracies.std()))