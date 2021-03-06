
# 5. Number of Instances: 150 (50 in each of three classes)

# 6. Number of Attributes: 4 numeric, predictive attributes and the class

# 7. Attribute Information:
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm
#    5. class: 
#       -- Iris Setosa
#       -- Iris Versicolour
#       -- Iris Virginica

# 8. Missing Attribute Values: None

%reset
# Load libraries
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Read data ---------------------------------
filename = 'D:/ml_projects/helloworld_irish/data/iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = read_csv(filename, names=names)


# Data exploration --------------------------------- 
print(df.sample(10))
print(df.info())
print(df.describe())
print(df.corr(method='pearson'))
print("NB null values = ",df.isnull().sum().max())
print(df.skew())

print(df.groupby('class').size()) # Apply function size group by class column

df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

df.hist()
pyplot.show()

# Scale data ----------------------------------------
# Scale all numeric column 
import numpy as np
from sklearn.preprocessing import StandardScaler

num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# Split test / train data set
array = df.values
type(array)
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Build model
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# Tune Hyperparameters for kNN ---------------------------------------------
# n_neighbors in [1 to 21]
# metric in [‘euclidean’, ‘manhattan’, ‘minkowski’]
# weights in [‘uniform’, ‘distance’]
# example of grid searching key hyperparametres for KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train, Y_train)

# summarize results
print("Best of kNN: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: 0.983333 using {'metric': 'euclidean', 'n_neighbors': 13, 'weights': 'uniform'}

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
	
# Tune param for logistic regression------------------------------------------
# full list of param https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = LogisticRegression()
c_values = [10000,5000,1000,500,200,150,100, 10, 1.0, 0.1, 0.01]
grid = dict(C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: 0.983333 using {'C': 1000}

# Neural netxork with Keras ------------------------------------------------
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Encode Categorical Variable (y)
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder   
le = LabelEncoder()
ohe = OneHotEncoder()

Y_train_ec = le.fit_transform(Y_train)
Y_train_ec = ohe.fit_transform(Y_train_ec.reshape(-1,1))

Y_validation_ec = le.fit_transform(Y_validation)
Y_validation_ec = ohe.fit_transform(Y_validation_ec.reshape(-1,1))

# Model : INPUT(4) => FC(8) => RELU => FC(8) => SOFTMAX
model = Sequential()
model.add(Dense(10, input_dim=4, init= "uniform" , activation= "relu"))
model.add(Dense(10, init= "uniform" , activation= "relu" ))
model.add(Dense(3, init= "uniform" , activation= "softmax" ))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train_ec , nb_epoch=150, batch_size=10)
scores = model.evaluate(X_train, Y_train_ec)
print('Keras accuracy: %.2f' % (scores[1]*100))

model.evaluate(X_train, Y_train_ec)
model.evaluate(X_validation, Y_validation_ec)









