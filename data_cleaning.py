# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:00:59 2021

@author: krish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data collected was saved to two excel sheets
df1 = pd.read_excel(r'C:\Users\krish\Music\Job_ML_project\final_data.xlsx', sheet_name='final_data', names=['Job_position', 'Company', 'Location', 'income', 'posting_time', 'requirements', 'rating', 'experience', 'link'], na_values=['#NAME?'], engine='openpyxl' )

df2 = pd.read_excel(r'C:\Users\krish\Music\Job_ML_project\software_dev.xlsx', sheet_name='new_jobs', names=['Job_position', 'Company', 'Location', 'income', 'posting_time', 'requirements', 'rating', 'experience', 'link'], na_values=['#NAME?'], engine='openpyxl' )

df = pd.concat([df1, df2])

# link & posting time for the jobs columns are not important for our analysis so we will drop them
df.drop('link', axis=1, inplace=True) 
df.drop('posting_time', axis=1, inplace=True) 

# Out of 22782 entries 22139 are duplicates, there were lot of duplicate values 
#looks like the company posted for the same profile many times after a gap of few days   
print('Length of Duplicated rows', len(df[df.duplicated()]))

# we can store this information in the column 'posting frequency'
# calculate posting frequency on the basis of company
freq = df[df.duplicated()]['Company'].value_counts()

# remove duplicates 
df.drop_duplicates(inplace=True)

# fill the frequency calculated
df['posting_frequency'] = df['Company'].map(freq)

# those not repeated will be null, therefore fill them as 1
df['posting_frequency'].fillna(1, inplace=True)


print('\n\n lets take a look at an example before calculating frequency: \n')
print(df[df['Company'] == 'BMC Software'])

# We just deleted duplicates but we still see multiple entries for some companies
# It looks like recently posted jobs with new tag are causing this, 
# lets remove them
df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew',''))

df.drop_duplicates(inplace=True)
df.index = np.arange(0,len(df))

# removing new line character from ratings
df['rating'] = df.rating.apply(lambda x: x.replace('\n',''))

# filling missing values with a value far away from our distribution
df['rating'].where(df['rating'] != 'na', 0, inplace=True)
df['rating'] = df['rating'].astype('float64')

# Rows with missing salaries contain valuable information regarding job position, location and their requirements
# So we will keep them for now 
# for now lets fill them with -999
df['Salary'].fillna('-999', inplace=True)

# remove new line and ruppes symbol  
df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('\n',''))
df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('₹',''))

df.to_csv('./data_cleaned.csv', index = False)
print('\n\n File Saved !!')