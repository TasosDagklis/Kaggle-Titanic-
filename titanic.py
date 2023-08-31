import pandas as pd
import numpy as np

#%%
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#%%
#subtraction of irrelevant and redundant variables
train = train_dataset.drop(['Name', 'Ticket', 'Cabin', 'Pclass'], axis=1)
test = test_dataset.drop(['Name', 'Ticket', 'Cabin', 'Pclass'], axis=1)

#%%
#Handling Outliers for the training set
#descriptive statistics
show = train.describe()
#η μεταβλητή fare έχει μεγάλη διαφορά στο 75% και στο max
#capping in the 95 persentile
train.loc[train.Fare > train.Fare.quantile(0.95), 'Fare'] = train.Fare.quantile(0.95)
test.loc[test.Fare > test.Fare.quantile(0.95), 'Fare'] = test.Fare.quantile(0.95)
#%%
#Handling Missing Values train set
mv_train = (train.isnull().sum() / len(train))
mv_test = (test.isnull().sum() / len(test))
#%%
#fix missing values
train.loc[train.Age.isnull(), 'Age'] = train.Age.mean()

test.loc[test.Age.isnull(), 'Age'] = test.Age.mean()
test.loc[test.Fare.isnull(), 'Fare'] = test.Fare.mean()
#%%
mv_train = (train.isnull().sum() / len(train))
mv_test = (test.isnull().sum() / len(test))
#%%
#Handling the Sex, train set
col = ['Sex', 'Embarked']
train = pd.get_dummies(train, columns=col)
test = pd.get_dummies(test, columns=col)
#%%
#spliting the train into x and y
X = train.drop(['Survived'], axis=1).reset_index(drop=True)
y = train['Survived'].reset_index(drop=True)
#%%
#spliting the data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
#Classification with K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

params = {
            'model__n_neighbors' : range(1,15),
            'model__weights': ['uniform', 'distance']
         }
knn = KNeighborsClassifier()
pipe = Pipeline([('scaler', MinMaxScaler()), ('model', knn)])
gridknn = GridSearchCV(estimator=pipe, param_grid=params)
gridknn.fit(X_train, y_train)
y_pred_knn = gridknn.predict(X_test)

#print the results
print("Best k: {:.1f}".format(gridknn.best_params_['model__n_neighbors']))
print("Best weighting: " + gridknn.best_params_['model__weights'])
print("Best grid score: {:.2f}".format(gridknn.best_score_))
print("K-NN Accuracy Score: {:.2f}".format(accuracy_score(y_test, y_pred_knn)))
#%%
#Classification with Logistic Regression
from sklearn.linear_model import LogisticRegression

params = {
            'model__C': np.logspace(-2, 3, 6) #try 6 values in logarithmic scale from 10^-2 to 10^3
        }

LogReg = LogisticRegression(max_iter=1000)
pipe = Pipeline([('scaler', MinMaxScaler()), ('model', LogReg)])
gridLogReg = GridSearchCV(estimator=pipe, param_grid=params)
gridLogReg.fit(X_train, y_train)
y_pred_reg = gridLogReg.predict(X_test)

#print the results
print("Best Lamda: {:.1f}".format(gridLogReg.best_params_['model__C']))
print("Best grid score: {:.2f}".format(gridLogReg.best_score_))
print("Logistic Regression Accuracy Score: {:.2f}".format(accuracy_score(y_test, y_pred_reg)))
#%%
#Classification with Extremely Randomized Trees
from sklearn.ensemble import ExtraTreesClassifier

trees = ExtraTreesClassifier(n_jobs=-1)
trees.fit(X_train, y_train)
y_pred_trees = trees.predict(X_test)

#print the results
print("Extra Trees Accuracy Score: {:.2f}".format(accuracy_score(y_test, y_pred_trees)))
#%%
#predict the test set
y_pred = gridknn.predict(test)
#%%
submission_file = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
submission_file.to_csv('submission.csv', index=False)