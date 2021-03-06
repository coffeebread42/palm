import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

### Input
filename = "flv2.out"

gamma = 0.001
cost = 10000

n_trails = 10
test_percentage = 0.33

### Process
data = np.genfromtxt(filename, delimiter=' ')

y = data[:, 0]
X = data[:, 1:]
o_a=[]
s_s=[]
s_p=[]

for t in range(n_trails):
    # Break the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_percentage, random_state = None)
    # Develop a model
    model = svm.SVC(C = cost, kernel='linear')
    model.fit(X_train, y_train)
    # Apply the model
    y_hat = model.predict(X_test)
    # Overall accuray
    oa = np.mean(y_test == y_hat)
    # Sensitivity = proportion of positive correctly classified
    ss = np.sum(np.logical_and(y_hat == 1, y_test == 1)) / np.sum(y_test == 1)
    # Specificity = proportion of negatives classified
    sp = np.sum(np.logical_and(y_hat == 0, y_test == 0)) / np.sum(y_test == 0)
    # Print results
    print(t, " ", oa*100, " ", ss*100, " ", sp*100)

    #append results
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

#print values
print(min_sp," ", max_sp," ",mean_sp," ",std_sp)
