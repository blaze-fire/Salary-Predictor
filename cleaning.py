# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:31:00 2021

@author: krish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./final_data.xlsx', sheet_name='final_data', names=['Job_position', 'Company', 'Location', 'Salary', 'posting_time', 'requirements', 'rating', 'experience', 'link'], na_values=['#NAME?'], engine='openpyxl' )

# link & posting time for the jobs columns are not important for our analysis so we will drop them
df.drop('link', axis=1, inplace=True) 
df.drop('posting_time', axis=1, inplace=True) 