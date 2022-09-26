import pandas as pd
import numpy as np
from utils.temp_clean_utils import Preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from pickle import dump
from sklearn.model_selection import GridSearchCV

# MAE b/c of outliers
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/data_train_preprocessed.csv')
df['experience'].fillna('', inplace=True)


"""#Calculating missing ratings"""

rating_df = df[df['avg_yearly_sal']>0]
rating_df.reset_index(drop=True, inplace=True)

#All rows with ratings present
rating_train = rating_df[~rating_df['rating'].isnull()]

#All rows with ratings absent
rating_test = rating_df[rating_df['rating'].isnull()]

train_set = rating_train.drop(['rating', 'avg_yearly_sal'], axis=1)
y_train = rating_train['rating']
test_set = rating_test.drop(['rating', 'avg_yearly_sal'], axis=1)


transformer = ColumnTransformer([ 
    ('vectorizer_job', TfidfVectorizer(), 'Job_position'), 
    ('vectorizer_comp', TfidfVectorizer(), 'Company'), 
    ('vectorizer_requirements', TfidfVectorizer(), 'requirements'),    
    ('vectorizer_exp', TfidfVectorizer(), 'experience')], remainder='passthrough')

train_set = transformer.fit_transform(train_set)
test_set = transformer.transform(test_set)

rating_model = XGBRegressor(random_state=42)
rating_model.fit(train_set, y_train)

test_ratings = rating_model.predict(test_set)


#add missing ratings to test dataset
rating_test['rating'] = list(np.round(test_ratings, 2))

#combing train and test to form dataset
final_df = pd.concat([rating_train, rating_test], axis=0)
final_df.sort_index(inplace=True)



"""#Create Dataset for Model building"""

X_train, X_test, y_train, y_test = train_test_split(final_df.drop('avg_yearly_sal', axis=1), final_df['avg_yearly_sal'], test_size = 0.01,random_state=42)
X_train.select_dtypes(include='object').columns

transformer = ColumnTransformer([ 
    ('vectorizer_job', TfidfVectorizer(), 'Job_position'), 
    ('vectorizer_comp', TfidfVectorizer(), 'Company'), 
    ('vectorizer_requirements', TfidfVectorizer(), 'requirements'),    
    ('vectorizer_exp', TfidfVectorizer(), 'experience')], remainder='passthrough')

train = transformer.fit_transform(X_train)
test = transformer.transform(X_test)

#Lasso Regression

from sklearn.linear_model import Lasso
lasso = Lasso(random_state=42)
param_grid = {'alpha': np.arange(1,101)/100, 'max_iter': [1000, 3000, 6000, 10000]} 
grid = GridSearchCV(lasso, param_grid=param_grid)
grid.fit(train, y_train)
lasso_best = grid.best_estimator_
pred = lasso_best.predict(test)

print(mean_absolute_error(y_test, pred))


# SVR

from sklearn.svm import SVR
svr = SVR()
param_grid = {'gamma': ['scale','auto'], 'C': [0.5, 1, 1.5]}
grid = GridSearchCV(svr, param_grid=param_grid)
grid.fit(train, y_train)
svr_best = grid.best_estimator_
pred = svr_best.predict(test)

print(mean_absolute_error(y_test, pred))

# Decision Tree

from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(criterion='mae', random_state=42)
dtree.fit(train, y_train)
pred = dtree.predict(test)

print(mean_absolute_error(y_test, pred))


# Random Forest
from sklearn.ensemble import RandomForestRegressor
param_grid = {'n_estimators' : [100, 300, 500]}
grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid)
grid.fit(train, y_train)
rnd_best = grid.best_estimator_
pred = rnd_best.predict(test)

print(mean_absolute_error(y_test, pred))


# XGBRegressor
xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(train, y_train)
pred = xgb_reg.predict(test)

print(mean_absolute_error(y_test, pred))




#train with complete dataset
#data_train = final_df.drop('avg_yearly_sal', axis=1)
#target = final_df['avg_yearly_sal']

#transformer = ColumnTransformer([
    #('vectorizer_job', TfidfVectorizer(), 'Job_position'),
    #('vectorizer_comp', TfidfVectorizer(), 'Company'),
    #('vectorizer_requirements', TfidfVectorizer(), 'requirements'),
    #('vectorizer_exp', TfidfVectorizer(), 'experience')], remainder='passthrough')

#text_data_transformed = transformer.fit_transform(data_train)

#xgb_reg = XGBRegressor(random_state=42)
#xgb_reg.fit(train_stack, target)


# # save the model
# dump(xgb_reg, open('model_building/model.pkl', 'wb'))
# # save the scaler
# dump(transformer, open('model_building/scaler.pkl', 'wb'))



# # load the model
# model = load(open('model.pkl', 'rb'))
# # load the scaler
# scaler = load(open('scaler.pkl', 'rb'))
