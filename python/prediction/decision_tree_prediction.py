import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy import stats

#data directory
inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf


### read data as np array
data = np.genfromtxt(filename, delimiter=' ')
#y is label column x is the variable columns
y = data[:, 0]
X = data[:, 1:]


# Break the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
# Develop a model
model = DecisionTreeClassifier(max_depth=5, min_samples_split=0.3, min_samples_leaf=0.3)
model.fit(X_train, y_train)
# Apply the model
y_hat = model.predict(X)


prediction = pd.DataFrame(y_hat, columns=['prediction']).to_csv('dt_p.csv')
#p = np.array([y_hat])
print(y_hat)
#print(p.T)
# Overall accuray
