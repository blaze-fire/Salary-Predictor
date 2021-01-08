# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:26:16 2021

@author: krish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\krish\Music\Job_ML_project\data_for_modelling.csv')  


# multiple linear regression
df_model = pd.get_dummies(df.drop(['Job_position', 'Company', 'Location', 'requirements','job_title', 'State'], axis=1))
