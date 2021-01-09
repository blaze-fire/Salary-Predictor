import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data_for_modelling.csv')

num_df = df.select_dtypes(exclude = 'object')

cat_df = df.select_dtypes(include = 'object')


corr = num_df.corr()

corr['avg_yearly_sal'].sort_values(ascending=False)[:-1]



