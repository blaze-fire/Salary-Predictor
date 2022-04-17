
import pandas as pd
import numpy as np
from utils.clean_utils import PreprocessOld, PreprocessNew

old_df = pd.read_csv('data/raw_data.csv')
old_df = PreprocessOld()(old_df)


new_df = pd.read_csv('data/final.csv')
new_df = PreprocessNew()(new_df)


final_df = pd.DataFrame()

for col in old_df.columns:
    temp_df = pd.concat([new_df[col], old_df[col]], axis=0)
    final_df = pd.concat([final_df, temp_df], axis=1)


from sklearn.utils import shuffle
df = shuffle(final_df)
df.reset_index(drop=True, inplace=True)




#here we will apply glove embeddings on all text columns and concatenate them and them concatenate them along with numerical columns
"""#Glove Embeddings"""

from utils.nlp_utils import Word2VecVectorizer
from gensim.models import KeyedVectors


# load GloVe model
filename = 'utils/glove_50d.gs'
model = KeyedVectors.load(filename, mmap='r')

def glove_embedded(X_train, _train, col, X_test=None, _test = None, test=False):
    
    if test:
      vectorizer = Word2VecVectorizer(model)
    
      X_train_embed = vectorizer.fit_transform(X_train[col].apply(str))
      X_test_embed = vectorizer.transform(X_test[col].apply(str))
      
      _train = np.concatenate((X_train_embed, _train), axis=1)
      _test = np.concatenate((X_test_embed, _test), axis=1)
      return _train, _test
      
    else:
        vectorizer = Word2VecVectorizer(model)
        X_embed = vectorizer.fit_transform(X_train[col].apply(str))
        _train = np.concatenate((X_embed, _train), axis=1)
     
        return _train

        

temp_df = df[df['avg_yearly_sal']>0]
temp_df.reset_index(drop=True, inplace=True)  


def calculate(df, col, col_list):
    train = df[~df[col].isnull()]
    test = df[df[col].isnull()]
    train_set = train.drop(col_list, axis=1)
    y_train = train[col]
    test_set = test.drop(col_list, axis=1)
    _train = train_set.select_dtypes(exclude='object').values
    _test = test_set.select_dtypes(exclude='object').values

    for col in test_set.select_dtypes(include='object').columns:
      _train, _test = glove_embedded(train_set, _train, col, test_set, _test, test=True)
      
    from xgboost import XGBRegressor
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(_train, y_train)
    missing_values = model.predict(_test)
    
    #add missing ratings to test dataset
    test['rating'] = np.round(missing_values, 2)
    
    #combing train and test to form dataset
    final_df = pd.concat([train, test], axis=0).sort_index()

    return final_df


"""#Calculating missing ratings"""

final_df = calculate(temp_df, 'rating', ['rating', 'avg_yearly_sal'])


"""#Calculating missing net experience"""

temp = calculate(temp_df, 'avg_yearly_sal', ['rating', 'avg_yearly_sal'])
final_df['net_experience'] = temp['net_experience']



"""#Transformation"""

from scipy import stats
import pylab
import matplotlib.pyplot as plt
#### Q-Q plot
def plot_data(df):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df.hist(bins=20)
    plt.subplot(1,2,2)
    stats.probplot(df,dist='norm',plot=pylab)
    plt.show()

temp_df = final_df.copy()

for col in temp_df.select_dtypes(exclude = 'object').columns:
  print(col, '\n')
  plot_data(final_df[col])
  temp_df[col], _ = stats.boxcox(1+final_df[col])
  plot_data(temp_df[col])





"""#Create Dataset for Model building"""

final_df = final_df[(~final_df['avg_yearly_sal'].isnull()) & (final_df['avg_yearly_sal'] > 0)]

final_df['avg_yearly_sal'] = final_df['avg_yearly_sal'].apply(lambda x: np.log(x))

plot_data(final_df['avg_yearly_sal'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df.drop('avg_yearly_sal', axis=1), final_df['avg_yearly_sal'], test_size = 0.01,random_state=42)

X = final_df.drop('avg_yearly_sal', axis=1)
y = final_df['avg_yearly_sal']

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)


#
train_ans = X_train.select_dtypes(exclude='object').values
test_ans = X_test.select_dtypes(exclude='object').values

for col in X_test.select_dtypes(include='object').columns:
  train_ans, test_ans = glove_embedded(X_train, train_ans, col, X_test, test_ans, test=True)




"***To train on entire dataset***"
train_ans = X.select_dtypes(exclude='object').values

for col in X.select_dtypes(include='object').columns:
  train_ans = glove_embedded(X, train_ans, col)






"""#Model """

#from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

#XGBoost
xgr = XGBRegressor(learning_rate = 0.1, max_depth = 5, random_state=42, objective = 'reg:squarederror', n_estimators = 800)
xgr.fit(train_ans, y_train)
pred = xgr.predict(test_ans)

np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))



#Train on entire dataset
xgr = XGBRegressor(learning_rate = 0.1, max_depth = 5, random_state=42, objective = 'reg:squarederror', n_estimators = 800)
xgr.fit(train_ans, y)

#import pickle 
#filename = 'xgb_model.sav'
#pickle.dump(xgr, open(filename, 'wb'))




