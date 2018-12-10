import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split


#data directory
inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf

gamma = 0.001
cost = 10000


### read data as np array
data = np.genfromtxt(filename, delimiter=' ')
#y is label column x is the variable columns
y = data[:, 0]
X = data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
model = svm.SVC(C = cost, kernel='linear')


model.fit(X_train, y_train)

model.score(X_test, y_test)

y_hat = model.predict(X)

prediction = pd.DataFrame(y_hat, columns=['prediction']).to_csv('svm_linear_p.csv')
#p = np.array([y_hat])
