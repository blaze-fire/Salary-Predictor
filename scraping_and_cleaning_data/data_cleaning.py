# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:00:59 2021

@author: krish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data collected was saved in a csv file

#df = pd.read_csv(r'C:\Users\krish\Music\Job_ML_project\data\raw_data.csv')

def clean_data(df):
    # link & posting time for the jobs columns are not important for our analysis so we will drop them
    df.drop('link', axis=1, inplace=True) 
    df.drop('posting_time', axis=1, inplace=True) 


    #Some company posted for the same profile many times after a gap of few days   
    # we can store this information in the column 'posting frequency'
    # calculate posting frequency on the basis of company
    freq = df[df.duplicated()]['Company'].value_counts()

    df.drop_duplicates(inplace=True)

    df['posting_frequency'] = df['Company'].map(freq)

    # those not repeated will be null, therefore fill them as 1
    df['posting_frequency'].fillna(1, inplace=True)

    # We just deleted duplicates but we still see multiple entries for some companies
    # It looks like recently posted jobs with new tag are causing this, 
    # lets remove them
    df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew',''))

    df.drop_duplicates(inplace=True)
    df.index = np.arange(0,len(df))


    df['rating'] = df.rating.apply(lambda x: str(x).replace('\n',''))
    df['rating'] = df['rating'].replace({"na":'0'})
    df['rating'] = df['rating'].astype('float64')
    df['rating'] = df['rating'].fillna(0)


    """
     Rows with missing salaries contain valuable information regarding job position, location and their requirements
     So we will keep them for now 
     for now lets fill them with -999
    """
    df['Salary'].fillna('-999', inplace=True)

    # remove new line and ruppes symbol  
    df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('\n',''))
    df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('₹',''))

    return df

#df = clean_data(df)


#df.to_csv(r'C:\Users\krish\Music\Job_ML_project\data\data_cleaned.csv', index = False)
#print('\n\n File Saved !!')
