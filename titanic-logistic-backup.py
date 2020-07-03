# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Prepare Titanic Model for Web / API
# 
# Final result on Heroku:
# https://titanic-model-app.herokuapp.com/
# 
# Sources:
# 1. https://www.kaggle.com/c/titanic/data
# 
# 2. https://towardsdatascience.com/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50?gi=30b632ffd17d
# 
# 2. https://blog.cambridgespark.com/deploying-a-machine-learning-model-to-the-web-725688b851c7
# 

# %%
get_ipython().run_cell_magic('html', '', '<style>\n  table {margin-left: 0 !important;}\n</style>\n# Align table to left')

# %% [markdown]
# ## Version Control
# | Version | Date | Person | Description         
# | :- |:------------- | :- | :-
# |1.00 |  2/3/2020 | SW| added version control and released to production
# |1.01 | 2/6/2020 | SW | added cross validation score 
# |1.02 | 2/8/2020 | SW | added hyperparameter tuning for logistic
# |1.03 | 2/16/2020 | SW | tested multiple models, including cross validation
# |1.04 | 2/26/2020 | SW | ran against test, exported submission, added feature importance and partial dependence to evaluate model
# |1.05 | 2/28/2020 | SW | added standard scaler to model and app
# |1.06 | 3/14/2020 | SW | updated feature importance graph 

# %%
# last updated
from datetime import datetime
print("last updated: " + str(datetime.now()))

# %% [markdown]
# ## Import modules / libraries and data

# %%
import pandas as pd # data manipulation
import seaborn as sns # for data exploration
import numpy as np

from sklearn.preprocessing import StandardScaler # for scaling inputs

from sklearn.linear_model import LogisticRegression # model: logistic regression
from sklearn.svm import SVC # model: SVC / support vector machine
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV # for hyperparameter tuning
from sklearn.model_selection import cross_val_score # for cross validation

from sklearn.inspection import plot_partial_dependence # for model evaluation and understanding
from pdpbox import pdp, get_dataset, info_plots # partial dependence 2 , may need need pip install pdpbox

import json # for exporting data


# %%
# create df,
data = pd.read_csv('titanic_train.csv')  # change file path as necessary

# %% [markdown]
# ## Data exploration, including creating sub-training set

# %%
data.shape


# %%
data.dtypes


# %%
data.describe(include='all') 


# %%
# see which is na
data.isna().sum()


# %%
# drop non-numeric & not relevant data
data = data.drop(['Name','Cabin','Embarked','Ticket'],axis = 1)
data.shape


# %%
# non Numeric data
# one-hot coding 
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore',sparse = False)

nonNumData = data.select_dtypes(exclude = ['int64','float64'])
print(nonNumData.dtypes)


# %%
# transformed data is ok because we've removed na's
nonNumDataEncArray = enc.fit_transform(nonNumData) 
nonNumDataEnc = pd.DataFrame(data = nonNumDataEncArray)


# %%
print(list(nonNumDataEnc.columns))
featuresNonNumeric = ['Sex-' + str(x) for x in list(nonNumDataEnc.columns)]
print(enc.categories_)
print(featuresNonNumeric)
nonNumDataEnc.columns = featuresNonNumeric


# %%
# add new data back to dataset (Sex-0 is 1 for females, Sex-1 is 1 for males)
data = pd.concat([data, nonNumDataEnc], axis=1)

# %% [markdown]
# ## Feature selection

# %%
# features and target
target = ['Survived']
features = ['Pclass', 'Age', 'SibSp', 'Fare'] + featuresNonNumeric # X matrix, y vector


# %%
# pairplot
allfeat = target + features
sns.pairplot(data, vars=allfeat)

# %% [markdown]
# ## Create and train model

# %%
# data is encoded with one-hot, ready for analysis
train = data


# %%
# drop null values
train.dropna(inplace=True)


# %%
# select data set for modeling
X = train[features]
X_orig = X

# scale inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = train[target].values.ravel() # model , covert to 1d array for easy fitting


# %%
#creat models for selection
logistic = LogisticRegression(solver='lbfgs',max_iter = 1000)
svc = SVC(gamma='auto')
randforest = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier()
gbc = GradientBoostingClassifier()
# svc = LinearSVC(max_iter = 1000) # doesn't work


# %%
model_set = [logistic, svc , randforest, knn, gbc]
model_set_results = {} 


# %%
#see which model scores best, using cross validation to remove overfitting
for mdl in model_set:
    scores = cross_val_score(mdl, X, y, cv=5)
    model_set_results[str(mdl)[0:20]] = np.mean(scores)


# %%
df = pd.DataFrame(model_set_results.items(), columns=["model", "score"])
print(df)

# %% [markdown]
# Gradient boost is best, accounting for cross validation

# %%
# logistic.get_params().keys()


# %%
# hyperparameter tuning after model selection
# # Create regularization penalty space
# penalty = ['l2']

# # Create regularization hyperparameter space
# C = [0.1,1,10]

# # Create hyperparameter options
# hyperparameters = dict(C=C, penalty=penalty)

# # Create grid search using 5-fold cross validation
# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)


# %%
#Conduct Grid Search

# # Fit grid search
# best_model = clf.fit(X, y)

# #View Hyperparameter Values Of Best Model

# # View best hyperparameters
# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
# print('Best C:',  best_model.best_estimator_.get_params()['C'])


# %%
# select model
model = gbc
model.fit(X,y)

# %% [markdown]
# ## Evaluate model

# %%
# roc curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y,model.predict(X) ) # roc_curve(actual, expected)
roc_auc = auc(fpr, tpr)


# %%
# plot roc curve
import matplotlib.pyplot as plt
plt.figure(figsize = [6.4*1.25, 4.8*1.25])
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# %%
#feature importance
model_importances_table = pd.DataFrame(data = {
    'Feature' : X_orig.columns , 
    'Importance' : model.feature_importances_
})

model_importances_table.sort_values(by='Importance', inplace=True, ascending=False)

plt.figure(figsize=(5, 10))
sns.barplot(x='Importance', y='Feature', data=model_importances_table)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.title('Gradient Boosting Classifier', size=10)

plt.show()
#Sex-0 is female


# %%
#partial dependence plot 1
pdp_features = ['Sex-0','Pclass','Age' ,('Sex-0','Pclass')] #selected
pdp_feature_names = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex-0', 'Sex-1'] #based on columns of X, sex-0 is female
plot_partial_dependence(model, X, features=pdp_features, feature_names = pdp_feature_names)


# %%
X_df = pd.DataFrame(data = X, columns = X_orig.columns)


# %%
# partial depedence plot detailed, from Kaggle explainability

# Create the data that we will plot
pdp_sex_0 = pdp.pdp_isolate(model=model, dataset=X_df, model_features=pdp_feature_names, feature='Sex-0')

# plot it
pdp.pdp_plot(pdp_sex_0, 'Sex-0', figsize = (10,5))
plt.show()


# %%
pdp_pclass = pdp.pdp_isolate(model=model, dataset=X_df, model_features=pdp_feature_names, feature='Pclass')

# plot it
pdp.pdp_plot(pdp_pclass, 'Pclass', figsize = (10,5))
plt.show()
#normalized x-axis bc is transformed


# %%
pdp_age = pdp.pdp_isolate(model=model, dataset=X_df, model_features=pdp_feature_names, feature='Age')

# plot it
pdp.pdp_plot(pdp_pclass, 'Age', figsize = (10,5))
plt.show()
#normalized x-axis bc is transformed

# %% [markdown]
# ## Run Model on Test Data for Kaggle Submission

# %%
data_test = pd.read_csv('test.csv')
print(data.dtypes)


# %%
data_test.describe(include='all')


# %%
print(featuresNonNumeric)


# %%
#add columns to dictionary 
data_test = data_test.assign(**dict.fromkeys(featuresNonNumeric, 0))


# %%
#update NonNumeric columns
data_test.loc[data_test['Sex'] == 'male', 'Sex-1'] = 1  # female 1 or male 1
data_test.loc[data_test['Sex'] == 'female', 'Sex-0'] = 0  # female 1 or male 1


# %%
X_predict = data_test[features]   # see feature selection above

#clean up na's
X_predict = X_predict.fillna(X_predict.mean())

#transform for fitting
X_predict = scaler.transform(X_predict)


# %%
# result for submission
Y_test = model.predict(X_predict)


# %%
# save submission
final_submission = pd.DataFrame({'PassengerID': data_test['PassengerId'], 'Survived': Y_test})


# %%
# export submission
final_submission.to_csv('final_submission.csv',index=False, header= True)

# %% [markdown]
# ## Save model and transformer as pickle

# %%
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# %% [markdown]
# ## Run app.py
# %% [markdown]
# type "python app.py" in console while in folder
# %% [markdown]
# ## Test app web interface
# Type "flask run" in terminal, once in this directory
# %% [markdown]
# ## Test app API
# 

# %%
# type "flask run" in command line while in parent directory (within anaconda env)

# local url
url = 'http://127.0.0.1:5000' 


# %%
# sample data
input_sample = {'Pclass': 3
      , 'Age': 2
      , 'SibSp': 1
      , 'Fare': 50}
input_sample = json.dumps(input_sample)


# %%
import requests, json

#send_request = requests.post(url, input_sample)
#print(send_request)


# %%
# check actual result
#print(send_request.json())

# %% [markdown]
# to stop app, press ctrl+c in console
# %% [markdown]
# ## Then create Procfile for Heroku app
# %% [markdown]
# ## Then create requirements.txt
# %% [markdown]
# Use:
# Flask==1.1.1
# gunicorn==19.9.0
# pandas==0.25.0
# requests==2.22.0
# scikit-learn==0.21.2
# scipy==1.3.1
# 
# More generally, can do:
# pip freeze > requirements.txt
# %% [markdown]
# ## Deploy on Heroku
# %% [markdown]
# ## Check Heroku

# %%
# heroku url
heroku_url = 'https://titanic-model-app.herokuapp.com/' # change to your app name# sample data
input_sample_api = {  'Pclass': 3
             , 'Age': 2
             , 'SibSp': 1
             , 'Fare': 50}
input_sample_api = json.dumps(input_sample_api)


# %%
# may need to disable firewall for this
#send_request = requests.post(heroku_url, data)
#print(send_request)


# %%
#print(send_request.json())

# %% [markdown]
# ## Exporting Options
# 1. From vscode, can export into jupyter notebook using vscode's jupyter interactive window
# 1. From Jupyter notebook/lab , can export notebook into html
# 1. From command line, can can convert using nbconvert to html ($ jupyter nbconvert --to FORMAT notebook.ipynb)
# 

# %%


