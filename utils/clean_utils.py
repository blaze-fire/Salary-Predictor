#Refactored code new time 18s from 220s
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import time

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

class Preprocess:

    
    def preprocess_text(self, text):
        #fast processing of stopwords from (https://stackoverflow.com/a/71095469/14204371)
        
        text = text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
        stopwords_dict = {word: 1 for word in stopwords.words("english")}
        text = " ".join([word for word in text.split() if word not in stopwords_dict])
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])
        return text


    def clean_data(self, df):

        df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew','').lower())
        df.replace('na', np.nan, inplace=True)
        df['rating'] = df['rating'].astype('float')

        return df



    # to calculate max and min Salary per annum
    def calc_salary(self, df):
        
        p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
        options = ['a year', 'a month', 'an hour']
        multiplyFactor = {'a year': 1, 'a month': 12, 'an hour':  9 * 22 * 12}
        
        df.Salary.fillna('', inplace=True)
        
        def get_salary(text):

            #to get whether the salary is given in per month or per year or per hour
            payplan = re.findall(r"(?=("+'|'.join(options)+r"))", text)
        
            if(payplan):
                sal = re.findall(p, text)
            
                #if range of salary is given we take their mean and use it as average salary
                if(len(sal) > 1):
                    minSal = int(re.sub(r'(?:(\d+?)),(\d+?)',r'\1\2',sal[0]))
                    maxSal = int(re.sub(r'(?:(\d+?)),(\d+?)',r'\1\2',sal[1]))
                    annualSal = ((minSal + maxSal)/2)*multiplyFactor[payplan[0]]
                    return annualSal
                
                else:
                    Sal = int(re.sub(r'(?:(\d+?)),(\d+?)',r'\1\2',sal[0]))
                    annualSal = Sal*multiplyFactor[payplan[0]]
                    return annualSal
                
            else:
                return 0
            
        df['avg_yearly_sal'] = df['Salary'].apply(get_salary)
        
        return df
    
    
    def work_location(self, df):

        def helper(text):
        
            if re.findall('full.time?', text.lower()):
                return 2
        
            elif re.findall('remote?', text.lower()):
                return 1
        
            else:
                return 0

        df['work_category'] = df['requirements'].fillna('').apply(helper)
        return df



    def calc_experience(self, df):
        
        def helper(text):

            if re.findall(r'\d+ year', text.lower()):
                exps = re.findall(r'\d+', text.lower())
                arr = np.array([int(x) for x in exps])
                return max(arr[arr<50])
        
            else:
                pass

        df['net_experience'] = df['experience'].fillna('').apply(helper).values

        return df



    #Educational criteria mentioned by these companies can also be useful
    def education(self, df):

        education_dict = {'bachelor':1, 'master':2, 'graduate':3}

        def helper(text):
            
            degree = re.findall(r'(graduate|bachelor|master)', text.lower())
            
            if degree:
                return education_dict[degree[0]]
        

        # Calculate education using experience and requirements
        # Then replace values in EduExp that are nans with values from EduReq

        EduExp = df['experience'].fillna('').apply(helper).values
        EduReq = df['requirements'].apply(helper).values
        result = np.where(np.isnan(EduReq), EduExp, EduReq)
        
        df['education_level'] = result
        return df




    def seniority(self, df):
        
        def helper(text):

            if re.findall(r'senior', text.lower()):
                return 2
        
            elif re.findall(r'junior', text.lower()):
                return 1
            
            else:
                return 0

        df['job_title'] = df['requirements'].apply(helper)

        return df


    def get_states(self, df):

        with open('utils/states.txt', 'r') as f:
            states = f.read()
            states_list = states.split(',')
        f.close()

        def helper(text):
            try:
                #Join the list on the pipe character |, which represents different options in regex.
                #Link: https://stackoverflow.com/a/33406382/14204371
                state = re.findall(r"(?=("+'|'.join(states_list)+r"))", text)[0]
                
                return states_list.index(state)
        
            except:
                return states_list.index('State_missing')

        df['State'] = df['Location'].apply(helper)
        return df



    def city(self, df):

        with open('utils/cities.txt', 'r') as f:
            cities = f.read()
            cities_list = cities.split(',')
        f.close()

        def get_city(text):
                  
            try:
                #Join the list on the pipe character |, which represents different options in regex.
                #Link: https://stackoverflow.com/a/33406382/14204371
                city = list(re.findall(r"(?=("+'|'.join(cities_list)+r"))", text)[0])[0]
                
                return cities_list.index(city)
            
            except:
                return cities_list.index('city_missing')

        df['Cities'] = df['Location'].apply(get_city)

        return df


    def final_operations(self, df):

        # remove columns with constant values
        df['Company'] = df['Company'].apply(lambda x: self.preprocess_text(str(x)))
        df['Job_position'] = df['Job_position'].apply(lambda x: self.preprocess_text(str(x)))
        
        df['requirements'] = df['requirements'].fillna('')
        df['requirements'] = df['requirements'].apply(lambda x: self.preprocess_text(str(x)))

        df['experience'] = df['experience'].fillna('')
        df['experience'] = df['experience'].apply(lambda x: self.preprocess_text(str(x)))
        df.drop(['Location', 'Salary'], axis=1, inplace=True)

        df = df.loc[:, (df != df.iloc[0]).any()]

        return df

    def __call__(self, df):
        start_time = time.time()

        df = self.clean_data(df)
        df = self.calc_salary(df)
        df = self.work_location(df)
        df = self.education(df)
        df = self.seniority(df)
        df = self.get_states(df)
        df = self.city(df)
        df = self.final_operations(df)

        print("--- Total time taken %s seconds ---" % (time.time() - start_time))
        
        return df
