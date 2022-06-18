import pandas as pd
import numpy as np
from utils.clean_utils import Preprocess
from utils.nlp_utils import Word2VecVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from gensim.models import KeyedVectors
from pickle import dump

#check
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





"""#Glove Embeddings"""
# load GloVe model
filename = 'utils/vector.kv'
model = KeyedVectors.load(filename)

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

rating_model = XGBRegressor(random_state=42)
rating_model.fit(rating_train, y_train)

test_ratings = rating_model.predict(rating_test)

#add missing ratings to test dataset
rating_test['rating'] = np.round(test_ratings, 2)

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

num_cols = list(X_train.select_dtypes(exclude='object').columns)
num_cols

train_stack = hstack((X_train[num_cols].values, train))
test_stack = hstack((X_test[num_cols].values, test))

xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(train_stack, y_train)
pred = xgb_reg.predict(test_stack)
print(np.sqrt(mean_squared_error(y_test, pred)))





#train with complete dataset
#data_train = final_df.drop('avg_yearly_sal', axis=1)
#target = final_df['avg_yearly_sal']

#transformer = ColumnTransformer([
    #('vectorizer_job', TfidfVectorizer(), 'Job_position'),
    #('vectorizer_comp', TfidfVectorizer(), 'Company'),
    #('vectorizer_requirements', TfidfVectorizer(), 'requirements'),
    #('vectorizer_exp', TfidfVectorizer(), 'experience')], remainder='passthrough')

#text_data_transformed = transformer.fit_transform(data_train)

#num_cols = list(data_train.select_dtypes(exclude='object').columns)
#num_cols

#train_stack = hstack((data_train[num_cols].values, text_data_transformed))

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




