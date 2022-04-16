
import pandas as pd
import numpy as np
from utils.temp_clean_utils import PreprocessOld, PreprocessNew


raw_df = pd.read_csv('data/old data/raw_data.csv')
len(raw_df)

raw_df = raw_df.iloc[:2000]


raw_df = PreprocessOld()(raw_df)

check_df = pd.read_csv('data/final.csv', names=['Job_position', 'Company', 'Salary', 'requirements', 'rating'])

check_df = PreprocessNew()(check_df)

#another_df = pd.read_csv('data/drive_data/final.csv')
#another_df = PreprocessNew()(another_df)

final_df = pd.DataFrame()

for col in check_df.columns:
    temp_df = pd.concat([check_df[col], raw_df[col]], axis=0)
    final_df = pd.concat([final_df, temp_df], axis=1)



from sklearn.utils import shuffle
df = shuffle(final_df)
df.reset_index(drop=True, inplace=True)


"""#Calculating missing ratings"""

temp_df = df[df['avg_yearly_sal']>0]
temp_df.reset_index(drop=True, inplace=True)
train = temp_df[~temp_df['rating'].isnull()]
test = temp_df[temp_df['rating'].isnull()]
train_set = train.drop(['rating', 'avg_yearly_sal'], axis=1)
y_train = train['rating']
test_set = test.drop(['rating', 'avg_yearly_sal'], axis=1)

"""#Glove Embeddings"""

from utils.nlp_utils import Word2VecVectorizer
from gensim.models import KeyedVectors

import gensim
gensim.__version__

# load GloVe model
filename = 'utils/glove_50d.gs'
model = KeyedVectors.load(filename, mmap='r')

rating_train = train_set.select_dtypes(exclude='object').values
rating_test = test_set.select_dtypes(exclude='object').values

def glove_embedded(X_train, col, X_test, rating_train, rating_test):

  vectorizer = Word2VecVectorizer(model)

  X_train_embed = vectorizer.fit_transform(X_train[col].apply(str))
  X_test_embed = vectorizer.transform(X_test[col].apply(str))
  
  rating_train = np.concatenate((X_train_embed, rating_train), axis=1)
  rating_test = np.concatenate((X_test_embed, rating_test), axis=1)
  
  return rating_train, rating_test


for col in test_set.select_dtypes(include='object').columns:
  rating_train, rating_test = glove_embedded(train_set, col, test_set, rating_train, rating_test)

from xgboost import XGBRegressor
rating_model = XGBRegressor(objective='reg:squarederror', random_state=42)
rating_model.fit(rating_train, y_train)

#import pickle 
#filename = 'rating_model.sav'
#pickle.dump(rating_model, open(filename, 'wb'))

test_ratings = rating_model.predict(rating_test)

#add missing ratings to test dataset
test['rating'] = np.round(test_ratings, 2)

#combing train and test to form dataset
final_df = pd.concat([train, test], axis=0)
final_df.sort_index(inplace=True)






"""#Calculating missing net experience"""

train = temp_df[~temp_df['net_experience'].isnull()]
test = temp_df[temp_df['net_experience'].isnull()]

train_indices = train.index.values
test_indices = test.index.values

train_set = train.drop(['net_experience', 'avg_yearly_sal'], axis=1)
y_train = train['net_experience']
test_set = test.drop(['net_experience', 'avg_yearly_sal'], axis=1)


#here we will apply glove embeddings on all text columns and concatenate them and them concatenate them along with numerical columns
"***Glove Embeddings***"

experience_train = train_set.select_dtypes(exclude='object').values
experience_test = test_set.select_dtypes(exclude='object').values

def glove_embedded(X_train, col, X_test, experience_train, experience_test):
  
  vectorizer = Word2VecVectorizer(model)

  X_train_embed = vectorizer.fit_transform(X_train[col].apply(str))
  X_test_embed = vectorizer.transform(X_test[col].apply(str))
  
  experience_train = np.concatenate((X_train_embed, experience_train), axis=1)
  experience_test = np.concatenate((X_test_embed, experience_test), axis=1)
  
  return experience_train, experience_test


for col in test_set.select_dtypes(include='object').columns:
  experience_train, experience_test = glove_embedded(train_set, col, test_set, experience_train, experience_test)


#will use xgboost to predict missing values
experience_model = XGBRegressor(objective= 'reg:squarederror', random_state=42)
experience_model.fit(experience_train, y_train)


#import pickle 
#filename = 'experience_model.sav'
#pickle.dump(experience_model, open(filename, 'wb'))

test_experience = experience_model.predict(experience_test)
test['net_experience'] = np.round(test_experience, 2)


temp = pd.concat([train, test], axis=0).sort_index()

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


final_df.to_csv('final_df.csv', index=False)


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


#here we will apply glove embeddings on all text columns and concatenate them and them concatenate them along with numerical columns
"***Glove Embeddings***"


train_ans = X_train.select_dtypes(exclude='object').values
test_ans = X_test.select_dtypes(exclude='object').values

def glove_embedded(X_train, col, X_test, train_ans, test_ans):
  
  vectorizer = Word2VecVectorizer(model)
  X_train_embed = vectorizer.fit_transform(X_train[col].apply(str))
  X_test_embed = vectorizer.transform(X_test[col].apply(str))
  train_ans = np.concatenate((X_train_embed, train_ans), axis=1)
  test_ans = np.concatenate((X_test_embed, test_ans), axis=1)
  
  return train_ans, test_ans


for col in X_test.select_dtypes(include='object').columns:
  train_ans, test_ans = glove_embedded(X_train, col, X_test, train_ans, test_ans)





"***To train on entire dataset***"
train_ans = X.select_dtypes(exclude='object').values

def glove_embedded(X, col,train_ans):
  
  vectorizer = Word2VecVectorizer(model)
  X_embed = vectorizer.fit_transform(X[col].apply(str))
  train_ans = np.concatenate((X_embed, train_ans), axis=1)
  
  return train_ans


for col in X.select_dtypes(include='object').columns:
  train_ans = glove_embedded(X, col,train_ans)



"""#Model """

#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#XGBoost

xgr = XGBRegressor(learning_rate = 0.02, max_depth = 7, random_state=42, objective = 'reg:squarederror', n_estimators = 10000)

xgr.fit(train_ans, y_train)



pred = xgr.predict(test_ans)
np.sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))


#Train on entire dataset
xgr = XGBRegressor(learning_rate = 0.02, max_depth = 7, random_state=42, objective = 'reg:squarederror', n_estimators = 10000)
xgr.fit(train_ans, y)

import pickle 
filename = 'xgb_word2vec.sav'
pickle.dump(xgr, open(filename, 'wb'))




