# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nlp_utils import preprocess_text, Word2VecVectorizer

df = pd.read_csv('data/raw_data.csv')


df.drop('link', axis=1, inplace=True)


def Salary(df):
    df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('\n',''))
    df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('₹',''))
    yearly_min = {}
    yearly_max = {}
    
    for i in range(0, len(df)):
        
        if df['Salary'][i] == '-999':
            yearly_min[i] = 0
            yearly_max[i] = 0
            
        if 'a year' in df['Salary'][i]:
            sal_min = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
            yearly_min[i] = float(sal_min)
            
            try:
                sal_max = df['Salary'][i].split('-')[1].replace('a year','').replace(',','')
                yearly_max[i] = float(sal_max)
                
            # if only single value present will be stored in both max and min, so the average comes accuate
            except:
                sal_max = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
                yearly_max[i] = float(sal_max)
            
       
        if 'a month' in df['Salary'][i]:
            sal_min = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
            yearly_min[i] = float(sal_min) * 12
            
            try:
                sal_max = df['Salary'][i].split('-')[1].replace('a month','').replace(',','')
                yearly_max[i] = float(sal_max) * 12    
                
            # if only single value present will be stored in both max and min, so the average comes accuate
            except:
                sal_max = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
                yearly_max[i] = float(sal_max) * 12
                
        
        if 'an hour' in df['Salary'][i]:
            sal_min = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','')
            yearly_min[i] = float(sal_min) * 9 * 22 * 12
            
            try:
                sal_max = df['Salary'][i].split('-')[1].replace('an hour','').replace(',','')
                yearly_max[i] = float(sal_max) * 9 * 22 * 12  
                
            # if only single value present will be stored in both max and min, so the average comes accuate
            except:
                sal_max = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','')
                yearly_max[i] = float(sal_max) * 9 * 22 * 12
    
    # min, max and avg Salary columns
    df['min_Salary'] = pd.DataFrame(yearly_min.values(), index= yearly_min.keys())
    df['max_Salary'] = pd.DataFrame(yearly_max.values(), index= yearly_max.keys())
    df['salary'] = ( df['min_Salary'] + df['max_Salary'] )/2
    df['monthly_Salary'] = df['salary']/12.

    df['salary'].fillna(-999, inplace=True)
    df['min_Salary'].fillna(-999, inplace=True)
    df['max_Salary'].fillna(-999, inplace=True)
    df['monthly_Salary'].fillna(-999, inplace=True)

    #Lets just drop these as we got our target column
    df.drop('max_Salary', axis=1, inplace=True)
    df.drop('min_Salary', axis=1, inplace=True)
    df.drop('monthly_Salary', axis=1, inplace=True)
    df.drop('Salary', axis=1, inplace=True)
    return df

df = Salary(df)

df = df[df['salary']>0]

df.reset_index(drop=True, inplace=True)



"""#ratings"""

df['rating'] = df['rating'].apply(lambda x: str(x).replace('\n',''))

df['rating'].replace(to_replace='na', value=None, inplace=True)

df['rating'] = df['rating'].astype(np.float64)

from scipy import stats
import pylab

#### Q-Q plot
def plot_data(df):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df.hist(bins=20)
    plt.subplot(1,2,2)
    stats.probplot(df,dist='norm',plot=pylab)
    plt.show()

plot_data(df['rating'].dropna())

temp = df.copy()

temp,parameters=stats.boxcox(df['salary'].dropna())
temp = pd.DataFrame(temp, columns=['salary'])
temp.reset_index(inplace=True)
#plot_data(temp)

plot_data(temp['salary'])
plot_data((df['salary'])**1/1.2)
df['salary'] = np.log(df['salary'])

df.isnull().sum()

def impute_nan(df,variable,median):
    df[variable+"_median"]=df[variable].fillna(median)
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'_random']=random_sample

median = df.rating.median()
impute_nan(df, 'rating', median)

plot_data(df['rating_random'])

df['rating_nan'] = np.where(df['rating'].isnull(),1,0)

df['rating_nan']


"""##Experience"""

def impute_nan_cat(df,variable):
    df[variable+"_newvar"]=np.where(df[variable].isnull(),"Missing",df[variable])

impute_nan_cat(df, 'experience')

df.drop('experience', axis=1, inplace=True)

cat_feats = df.drop(['salary', 'posting_time','requirements'], axis=1).select_dtypes(include='object').columns

cat_feats

"""##model training"""

text = df['requirements']

clean_text = preprocess_text(text)

df['clean_text']=clean_text

temp = df.drop('rating', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(temp.drop(['salary', 'posting_time','requirements'], axis=1), df['salary'], random_state=42)

X_train.reset_index(drop=True, inplace=True)




"""#Glove Embeddings"""

from gensim.models import KeyedVectors
# load GloVe model
filename = 'utils/word2vec_model.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

vectorizer = Word2VecVectorizer(model)

num_train = X_train.select_dtypes(exclude = 'object')
cat_train = X_train.select_dtypes(include = 'object')


train_ans = X_train.select_dtypes(exclude='object').values
test_ans = X_test.select_dtypes(exclude='object').values

def glove_embedded(X_train, col, X_test, train_ans, test_ans):
  X_train_embed = vectorizer.fit_transform(X_train[col])
  X_test_embed = vectorizer.transform(X_test[col])
  train_ans = np.concatenate((X_train_embed, train_ans), axis=1)
  test_ans = np.concatenate((X_test_embed, test_ans), axis=1)
  return train_ans, test_ans


for col in X_test.select_dtypes(include='object').columns:
  train_ans, test_ans = glove_embedded(X_train, col, X_test, train_ans, test_ans)



"""#Feature Selection"""

from sklearn.feature_selection import VarianceThreshold

var_thres = VarianceThreshold(threshold=0)
var_thres.fit(train_ans)

np.unique(var_thres.get_support())

from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

xgb_reg = XGBRegressor()

xgb_reg.fit(train_ans, y_train)

model = SelectFromModel(xgb_reg, prefit=True)

X_train_new = model.transform(train_ans)
X_test_new = model.transform(test_ans)

X_test_new.shape
X_train_new.shape


"""## Random Forest"""

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(oob_score=True)
rf_reg.fit(X_train_new, y_train)


from sklearn.metrics import mean_squared_error
pred = rf_reg.predict(X_test_new)

np.sqrt(mean_squared_error(y_test, pred))



from sklearn.model_selection import GridSearchCV

params = {'n_estimators':[200], 'min_samples_split':[3, 4, 5]}
grid_search = GridSearchCV(rf_reg, params, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train_new, y_train)


np.sqrt(np.abs(grid_search.best_score_))
grid_search.best_params_
best_model = grid_search.best_estimator_


import pickle
filename = 'finalized_random_forest.sav'
pickle.dump(best_model, open(filename, 'wb'))



"""model training"""

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

lasso = Lasso(max_iter=2000)

lasso_scores = cross_val_score(lasso, train_ans, y_train, scoring = 'neg_mean_squared_error')

np.exp(np.sqrt(np.abs(lasso_scores)))

lasso.fit(train_ans, y_train)
pred = lasso.predict(test_ans)
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



from xgboost import XGBRegressor
xgr = XGBRegressor()
scores = cross_val_score(xgr, train_ans, y_train, scoring = 'neg_mean_squared_error')
np.mean(np.sqrt(np.abs(scores)))
pred = xgr.predict(test_ans)
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(train_ans, y_train)
pred = rf_reg.predict(test_ans)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



from sklearn.svm import SVR
svr = SVR()
svr.fit(train_ans, y_train)
pred = svr.predict(test_ans)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


from sklearn.ensemble import VotingRegressor
vot_reg = VotingRegressor(estimators=[('rf', rf_reg), ('xgr', xgb_reg)])
vot_reg.fit(X_train_new, y_train)
pred = vot_reg.predict(X_test_new)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



import pickle
filename = 'all_trained_models/finalized_model.sav'
pickle.dump(vot_reg, open(filename, 'wb'))

