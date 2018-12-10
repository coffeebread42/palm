import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
model = RandomForestClassifier(max_depth= 7,
                                n_estimators= 500,
                                min_samples_leaf= 0.2,
                                min_samples_split= 0.58)

model.fit(X_train, y_train)

model.score(X_test, y_test)

y_hat = model.predict(X)

prediction = pd.DataFrame(y_hat, columns=['prediction']).to_csv('rf_p2.csv')
#p = np.array([y_hat])
