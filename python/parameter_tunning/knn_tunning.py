from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

inf ="/home/roberto/Desktop/work/src/classifiers/flv2.out"
### Input
filename = inf

#gamma = 0.001
#cost = 10000
n_trails = 10

#test_percentage = 0.33

### Process
data = np.genfromtxt(filename, delimiter=' ')

y = data[:, 0]
X = data[:, 1:]
o_a=[]
s_s=[]
s_p=[]

#break data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
#Develop model
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# #apply model
# model.score(X_test, y_test)
# #predict
# y_hat = model.predict(X_test)

#AUC_evaluation metric
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print("AUC of default model is:")
# print(roc_auc)

#finding amount of nieghbors
#give range for neighbors to iterate thru
"""neighbors = list(range(1,30))
#make list to fill with data from loop
train_results = []
test_results = []
#loop to search right amount off neighbors
for n in neighbors:
   model = KNeighborsClassifier(n_neighbors=n)
   model.fit(X_train, y_train)
   train_pred = model.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)



line1, = plt.plot(neighbors, train_results, 'b', label="Train AUC")
line2, = plt.plot(neighbors, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_neighbors')
plt.savefig("n_neighbors_optimal.png")
plt.show()"""

#finding the distance parameters
"""distances =[1,2,3,4,5]
train_results = []
test_results = []
for p in distances:
   model = KNeighborsClassifier(p=p)
   model.fit(X_train, y_train)

   train_pred = model.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   y_hat = model.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)



line1, = plt.plot(distances, train_results, 'b', label="Train AUC")
line2, = plt.plot(distances, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('p')
plt.savefig("p_in_distance.png")
plt.show()"""

for t in range(n_trails):
#break data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = None)
#Develop model
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
#apply model
    model.score(X_test, y_test)

    y_hat = model.predict(X_test)

    # Overall accuray
    oa = np.mean(y_test == y_hat)

    # Sensitivity = proportion of positive correctly classified
    ss = np.sum(np.logical_and(y_hat == 1, y_test == 1)) / np.sum(y_test == 1)

    # Specificity = proportion of negatives classified
    sp = np.sum(np.logical_and(y_hat == 0, y_test == 0)) / np.sum(y_test == 0)

    print(t, " ", oa*100, " ", ss*100, " ", sp*100)
    o_a.append(oa)
    s_s.append(ss)
    s_p.append(sp)
#transform list into array
oa1=np.array(o_a)
ss1=np.array(s_s)
sp1=np.array(s_p)
#descriptive stats for oa
min_oa = oa1.min()
max_oa = oa1.max()
mean_oa = oa1.mean()
std_oa = oa1.std()

#descriptive stats for ss
min_ss = ss1.min()
max_ss = ss1.max()
mean_ss = ss1.mean()
std_ss = ss1.std()

#descriptive stats for sp
min_sp = sp1.min()
max_sp = sp1.max()
mean_sp = sp1.mean()
std_sp = sp1.std()

#print values oa
print(min_oa," ", max_oa," ",mean_oa," ",std_oa)


#print values ss
print(min_ss," ", max_ss," ",mean_ss," ",std_ss)

#print values sp
print(min_sp," ", max_sp," ",mean_sp," ",std_sp)

    #print("OA = ", oa * 100)
    #print("SS = ", ss * 100)
    #print("SP = ", sp * 100)
