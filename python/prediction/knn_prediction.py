from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#data directory
inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf


### read data as np array
data = np.genfromtxt(filename, delimiter=' ')
#y is label column x is the variable columns
y = data[:, 0]
X = data[:, 1:]





#break data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
#Develop model
model = KNeighborsClassifier(n_neighbors=3, p=3)
model.fit(X_train, y_train)
#apply model
model.score(X_test, y_test)

y_hat = model.predict(X)

prediction = pd.DataFrame(y_hat, columns=['prediction']).to_csv('knn_p.csv')
#p = np.array([y_hat])
print(y_hat)
#print(p.T)
# Overall accuray
