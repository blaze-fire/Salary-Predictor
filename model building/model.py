
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

df = pd.read_csv('./data/data_for_modelling.csv')

num_df = df.select_dtypes(exclude = 'object')


# Log transform turns out to be most suitable for our distribution as the other two scaling methods failed to deal with outliers effectively 


num_df['avg_yearly_sal'] = num_df['avg_yearly_sal'].apply(lambda x: np.log(x) if x>0 else 0)

sns.displot(num_df['avg_yearly_sal'], kde=True)
# Now our annual salary distribution is quite uniform

# Lets do the train test split
#  Note  : Here we are using train test split as dataset is quite small but if you have a much bigger dataset you might want to consider using stratified shuffle split.
# In our case train test split gave better results compared to stratified shuffling


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(num_df.drop('avg_yearly_sal', axis=1), num_df['avg_yearly_sal'], test_size=100, random_state=42)


from sklearn.ensemble import RandomForestRegressor, VotingRegressor


rnd_reg = RandomForestRegressor(oob_score=True, random_state=42)
rnd_reg.fit(X_train , y_train)



col_names = num_df.drop('avg_yearly_sal', axis=1).columns


# plot to see top 10 important features by random forest
plt.figure(figsize=(8,15))
importances = rnd_reg.feature_importances_ [:10]
idxs = np.argsort(importances) 
plt.title('Feature Importances') 
plt.barh(range(len(idxs)), importances[idxs], align='center') 
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs]) 
plt.xlabel('Random Forest Feature Importance') 
plt.show() 


x = [col_names[i] for i in idxs][::-1]

cols = ['rating','net_experience', 'jr', 'senior', 'bachelor', 'masters', 'posting_frequency']

X_train = X_train[cols]

X_test = X_test[cols]



rnd_reg.oob_score_

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# Lets tune our random forest for better performance


param_grid = {'n_estimators' : [100, 300, 500]}
grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid)
grid.fit(X_train, y_train)
rnd_best = grid.best_estimator_
pred = rnd_best.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


filename = './all_trained_models/rnd_best.sav'
pickle.dump(rnd_best, open(filename, 'wb'))


# Now we will train and fine tune many models namely Lasso regression, Decision tree, Random Forest, Extra trees, Gradient Boosted trees, Xgboost and then using Voting regressor on best performing models


from sklearn.svm import SVR
svr = SVR()
param_grid = {'gamma': ['scale','auto'], 'C': [0.5, 1, 1.5]}
grid = GridSearchCV(svr, param_grid=param_grid)
grid.fit(X_train, y_train)
svr_best = grid.best_estimator_
pred = svr_best.predict(X_test)

grid.best_params_

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))

filename = './all_trained_models/svr_best.sav'
pickle.dump(svr_best, open(filename, 'wb'))




from sklearn.linear_model import Lasso
lasso = Lasso(random_state=42)
param_grid = {'alpha': np.arange(1,101)/100, 'max_iter': [1000, 3000, 6000, 10000]} 
grid = GridSearchCV(lasso, param_grid=param_grid)
grid.fit(X_train, y_train)
lasso_best = grid.best_estimator_
pred = lasso_best.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


filename = './all_trained_models/lasso_best.sav'
pickle.dump(lasso_best, open(filename, 'wb'))


from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(criterion='mae', random_state=42)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


filename = './all_trained_models/DecisionTree.sav'
pickle.dump(dtree, open(filename, 'wb'))

from xgboost import XGBRegressor
xgr = XGBRegressor(random_state=42)
xgr.fit(X_train, y_train)
pred = xgr.predict(X_test)



np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



param_grid = {"learning_rate"    : [0.05, 0.20, 0.30 ] , "max_depth"        : [ 3, ],"gamma"            : [ 0.1, 0.4 ]}



grid = GridSearchCV(xgr, param_grid=param_grid)
grid.fit(X_train, y_train)
xgr_best = grid.best_estimator_
pred = xgr_best.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



filename = './all_trained_models/Xgr_best.pkl'
pickle.dump(xgr_best, open(filename, "wb"))




from sklearn.ensemble import ExtraTreesRegressor
extra_reg = ExtraTreesRegressor(n_estimators=500, random_state=42)
extra_reg.fit(X_train, y_train)
pred = extra_reg.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



filename = './all_trained_models/Extra_best.sav'
pickle.dump(extra_reg, open(filename, 'wb'))



from sklearn.ensemble import GradientBoostingRegressor
grb_reg = GradientBoostingRegressor(loss='lad', random_state=42)
grb_reg.fit(X_train, y_train)
pred = grb_reg.predict(X_test)



np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



filename = './all_trained_models/GB_best.sav'
pickle.dump(grb_reg, open(filename, 'wb'))


# Now using Voting regressor to train on the best performing models so far, which averages the individual prediction to form a final prediction.


vot_reg = VotingRegressor(estimators=[('rf', rnd_best), ('xg', xgr_best), ('gb', grb_reg), ('eg', extra_reg)])

vot_reg.fit(X_train, y_train)
pred = vot_reg.predict(X_test)


np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


filename = './all_trained_models/voting_best.sav'
pickle.dump(vot_reg, open(filename, "wb"))


# KNN
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()


param_grid = {"n_neighbors"    : [3, 5, 9, 11] , "weights"        : [ 'uniform', 'distance' ], "metric"            : [ 'euclidean', 'manhattan']}

grid = GridSearchCV(knn, param_grid=param_grid)
grid.fit(X_train, y_train)
knn_best = grid.best_estimator_
pred = knn_best.predict(X_test)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


filename = './all_trained_models/knn_best.sav'
pickle.dump(vot_reg, open(filename, "wb"))


# We can see that we have used some really powerful models like Random forest, ExtraTrees, Gradient boosted trees and Xgboost models as the complexity of problem is high but the available data is small. (784 training and 100 test examples) 
# Lets go one step further and create a blender of best models so far, to squeeze a bit more performance from our models
X_test[0, :].shape
# models to use in our blender 
estimators = [rnd_best, dtree]
X_test[0,:].reshape(1,-1).shape

ans=[]
for reg in estimators:
    pred = reg.predict(X_test)
    ans.append(pred)

ans = sum(ans)/len(ans)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(ans)))

X_train_predictions = np.empty((len(X_train), len(estimators)), dtype = np.float32)


for index, estimator in enumerate(estimators):
    X_train_predictions[:, index] = estimator.predict(X_train)


rnd_reg1 = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=42)


rnd_reg1.fit(X_train_predictions, y_train)


X_test_predictions = np.empty((len(X_test), len(estimators)), dtype = np.float32)


for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

pred = rnd_reg1.predict(X_test_predictions)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))

from sklearn.metrics import r2_score

r2_score(y_test, pred)


# calculate 95 percent confidence interval for our predictions
from scipy import stats
confidence = 0.95
squared_errors = (np.exp(ans) - np.exp(y_test)) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 
                        loc=squared_errors.mean(), 
                        scale=stats.sem(squared_errors))) 
# Our model explains half of the observed variation, which is great.   
# We can also conclude that the model can give much better predictions if fed with more data.





