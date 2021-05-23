# -*- coding: utf-8 -*-

pip -q install catboost

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline 

from datetime import datetime 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

data_df = pd.read_csv('data.csv')

data_df.head()

data_df.describe(include='all')

"""Handling Missing Values"""

total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
total = pd.DataFrame(total, columns=['Missing Values'])
total['Columns'] = total.index
total.reset_index(drop=True, inplace=True)

percent = pd.DataFrame(percent, columns=['Percentage Missing Values'])
percent['Columns'] = percent.index
percent.reset_index(drop=True, inplace=True)
total.merge(percent, left_on='Columns', right_on='Columns')[['Columns','Missing Values','Percentage Missing Values' ]]

# Imputing the Loan Amount with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data_df[['LoanAmount']])
data_df[['LoanAmount']] = imputer.fit_transform(data_df[['LoanAmount']])

#dropping rows where the we have null values in 'Married', 'Gender', 'Loan_Amount_Term', 'Dependents'
data_df.dropna(subset=['Married', 'Gender', 'Loan_Amount_Term', 'Dependents'], inplace = True)

# imputing missing values with the mode of that column
# for each column, get value counts in decreasing order and take the index (value) of most common class
data_df = data_df.apply(lambda x: x.fillna(x.value_counts().index[0]))

pd.DataFrame(data_df.isnull().sum(), columns=['Missing Value'])

"""Target Variable"""

plt.figure(figsize=[6,6])
temp = data_df["Loan_Status"].value_counts()
df = pd.DataFrame({'Class': temp.index,'Values': temp.values})
ax = sns.barplot('Class', 'Values', data = df,  palette='hls')
plt.title('Target Variable')

"""There 192 Nos and 422 Ys, approximately 31% of the values are No, which means that only 31% of the time the loan is not approved. That means the data is unbalanced with respect with target variable Class.

EDA
"""

plt.figure(figsize=[6,6])
sns.jointplot(x ="ApplicantIncome", y ="CoapplicantIncome", data = data_df ,palette='hls', hue = "Loan_Status")

plt.figure(figsize=[6,6])
sns.histplot(data=data_df, x="LoanAmount", kde=True,palette='hls', bins=30, hue = 'Loan_Status')

plt.figure(figsize=[6,6])
sns.countplot(x = 'Education', data = data_df, hue='Loan_Status', palette='hls')

plt.figure(figsize=[6,6])
sns.histplot(data=data_df, x="Loan_Amount_Term", kde=True,palette='hls', bins=5, hue = 'Loan_Status')

# seaborn style
sns.set(style = "ticks", palette='hls')
plt.figure(figsize=[6,6])
# creates FacetGrid
g = sns.FacetGrid(data_df, col = "Gender", row = 'Married', palette='hls')
g.map(plt.hist, "Loan_Status");

plt.figure(figsize=[6,6])
g = sns.FacetGrid(data_df, col = "Education", hue = "Loan_Status",  palette='hls' )
g.map(plt.scatter, "ApplicantIncome", "CoapplicantIncome", alpha =.7)
g.add_legend();

plt.figure(figsize=[6,6])
sns.countplot('Property_Area', data = data_df, hue = 'Loan_Status', palette='hls')

plt.figure(figsize=[6,6])
sns.countplot('Credit_History', data = data_df, hue = 'Loan_Status', palette='hls')

"""Encoding"""

# Encoding Target Variable
le = LabelEncoder()
data_df['Loan_Status'] = le.fit_transform(data_df['Loan_Status'])

# Encoding Categorical Variable
data_df = pd.get_dummies(data_df,
                         prefix= ['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area','Loan_Amount_Term'], 
                         columns= ['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area', 'Loan_Amount_Term'], 
                         drop_first=True)

"""Droping Loan_ID"""

# Dropping LoanID
data_df.drop('Loan_ID', axis = 1, inplace=True)

"""Correlation Plot"""

plt.figure(figsize=[10,10])
sns.heatmap(data_df.corr(),vmax=1, vmin=-1 )

"""### Splitting Data"""

X = data_df.drop(['Loan_Status'], axis = 1)
y = data_df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""Feature Selection"""

clf = RandomForestClassifier(n_jobs=4, 
                             random_state=42,
                             criterion='gini',
                             n_estimators=100,
                             verbose=False)
clf.fit(X_train,y_train)

tmp = pd.DataFrame({'Feature': X.columns, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp, palette='hls')
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()

features = tmp['Feature'][:8].to_list()
features

X_train, X_test = X_train[features], X_test[features]

"""### Random Forest"""

clf = RandomForestClassifier(n_jobs=4, 
                             random_state=42,
                             criterion='gini',
                             n_estimators=100,
                             verbose=False, 
                             max_depth=20, 
                             min_samples_split=2, 
                             max_features='auto'
                             ) 
clf.fit(X_train,y_train)

roc_auc_score(y_test, clf.predict(X_test)), f1_score(y_test, clf.predict(X_test)), accuracy_score(y_test, clf.predict(X_test))

confusion_matrix(y_test, clf.predict(X_test))

"""### Gradient Boosting"""

from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(
    warm_start= True, 
    learning_rate=0.2, 
    n_estimators=100, 
    min_samples_split=3,
    random_state=0, 
    max_depth =9).fit(X_train, y_train)

roc_auc_score(y_test, gbt.predict(X_test)), f1_score(y_test, gbt.predict(X_test)), accuracy_score(y_test, gbt.predict(X_test))

confusion_matrix(y_test, clf.predict(X_test))

"""# AdaBoostClassifier"""

ada = AdaBoostClassifier(random_state=42,
                         algorithm='SAMME.R',
                         learning_rate=.9,
                         n_estimators=500)
ada.fit(X_train,y_train)

roc_auc_score(y_test, ada.predict(X_test)), f1_score(y_test, ada.predict(X_test)), accuracy_score(y_test, ada.predict(X_test))

"""# CatBoostClassifier"""

cat = CatBoostClassifier(iterations=500,
                             learning_rate=0.02,
                             depth=12,
                             eval_metric='AUC',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=100)
cat.fit(X_train,y_train)
roc_auc_score(y_test, cat.predict(X_test)), f1_score(y_test, cat.predict(X_test)), accuracy_score(y_test, cat.predict(X_test))

"""# LGB"""

params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.05,
          'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 4,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':150, # because training data is extremely unbalanced 
         }

dtrain = lgb.Dataset(X_train.values, 
                     label=y_train.values,)

dtest = lgb.Dataset(X_test.values,
                     label=y_test.values)

evals_results = {}

model = lgb.train(params, 
                  dtrain, 
                  valid_sets=[dtrain, dtest], 
                  valid_names=['train','valid'], 
                  evals_result=evals_results, 
                  num_boost_round=1000,
                  early_stopping_rounds=2*50,
                  verbose_eval=50, 
                  feval=None)

preds = model.predict(X_test)

roc_auc_score(y_test, preds)

"""# Standardization of Numerical Variables"""

sc = StandardScaler()
X_train[['ApplicantIncome','LoanAmount','CoapplicantIncome']] = sc.fit_transform(X_train[['ApplicantIncome','LoanAmount','CoapplicantIncome']])
X_test[['ApplicantIncome','LoanAmount','CoapplicantIncome']] = sc.fit_transform(X_test[['ApplicantIncome','LoanAmount','CoapplicantIncome']])

"""Logistic Regression"""

lr = LogisticRegression(penalty='l2', max_iter=500).fit(X_train,y_train)
roc_auc_score(y_test, lr.predict(X_test)), f1_score(y_test, lr.predict(X_test)), accuracy_score(y_test, lr.predict(X_test))

"""# K Nearest Neighbors"""

from sklearn.neighbors import KNeighborsClassifier 
for i in [1,4,8,10,16,20,50]:
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    print(i, roc_auc_score(y_test, knn.predict(X_test)), f1_score(y_test, knn.predict(X_test)), accuracy_score(y_test, knn.predict(X_test)))

"""Neural Network"""

# import library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# neural net formation
model = Sequential()
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose = 1, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

