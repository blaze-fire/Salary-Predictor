import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


class PreprocessOld:

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = ' '.join([word for word in text.split() if not word in set(stopwords.words('english'))])
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

        return text
    
    def clean_data(self, df):
        #Some company posted for the same profile many times after a gap of few days   
        # we can store this information in the column 'posting frequency'
        # calculate posting frequency on the basis of company
        freq = df['Company'].value_counts()

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['posting_frequency'] = df['Company'].map(freq)

        # those not repeated will be null, therefore fill them as 1
        df['posting_frequency'].fillna(1, inplace=True)

        # We just deleted duplicates but we still see multiple entries for some companies
        # It looks like recently posted jobs with new tag are causing this, 
        # lets remove them
        df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew','').lower())

        df.replace('na', np.nan, inplace=True)
        df['rating'] = df['rating'].astype('float')


        """
         Rows with missing salaries contain valuable information regarding job position, location and their requirements
         So we will keep them for now 
         for now lets fill them with -999
        """

        df['Salary'].fillna('-999', inplace=True)

        # remove new line and ruppes symbol  
        df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('\n','').replace('₹',''))

        return df



    # to calculate max and min Salary per annum
    def calc_salary(self, df):

        yearly_min = {}
        yearly_max = {}

        for i in range(0, len(df)):

            if df['Salary'][i] == '-999':
                yearly_min[i] = 0
                yearly_max[i] = 0

            if 'a year' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
                yearly_min[i] = int(sal_min)

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('a year','').replace(',','')
                    yearly_max[i] = int(sal_max)

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
                    yearly_max[i] = int(sal_max)


            if 'a month' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
                yearly_min[i] = int(sal_min) * 12

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('a month','').replace(',','')
                    yearly_max[i] = int(sal_max) * 12    

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
                    yearly_max[i] = int(sal_max) * 12


            if 'an hour' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','').replace(' ','')
                yearly_min[i] = float(sal_min) * 9 * 22 * 12

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('an hour','').replace(',','').replace(' ','')
                    yearly_max[i] = float(sal_max) * 9 * 22 * 12  

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','').replace(' ','')
                    yearly_max[i] = float(sal_max) * 9 * 22 * 12

        # min, max and avg Salary columns
        df['min_Salary'] = pd.DataFrame(yearly_min.values(), index= yearly_min.keys())
        df['max_Salary'] = pd.DataFrame(yearly_max.values(), index= yearly_max.keys())
        df['avg_yearly_sal'] = ( df['min_Salary'] + df['max_Salary'] )/2.

        #Lets just drop these as we got our target column
        df.drop(['max_Salary', 'min_Salary', 'Salary'], axis=1, inplace=True)

        df['avg_yearly_sal'].fillna(0, inplace=True)

        return df



    def work_location(self, df):

        work_type = []
        for j in range(len(df)):

            if re.findall('full.time?', df['requirements'][j].lower()):
                work_type.append(2)

            elif re.findall('remote?', df['requirements'][j].lower()):
                work_type.append(1)

            else:
                work_type.append(0)


        df['work_category'] = work_type

        return df



    def calc_experience(self, df):
        df['experience'].fillna('', inplace = True)

        experience_list = []
        for j in range(len(df)):

            if re.findall(r'\d+ year', df['experience'][j]):
                experience_list.append(re.search(r'\d+ year', df['experience'][j]).group()[0])

            else:
                experience_list.append('')

        df['net_experience'] = experience_list
        df['net_experience'] = pd.to_numeric(df['net_experience'])
                
        return df



    #Educational criteria mentioned by these companies can also be useful
    def education(self, df):

        education_list = []
        education_dict = {'bachelor':1, 'master':2, 'graduate':3}
        for j in range(len(df)):
            
            if re.findall(r'(graduate|bachelor|master)', df['experience'][j].lower()):
                education_list.append( education_dict[re.search(r'(graduate|bachelor|master)', df['experience'][j].lower()).group()] )
            
            elif re.findall(r'(graduate|bachelor|master)', df['requirements'][j].lower()):
                education_list.append( education_dict[re.search(r'(graduate|bachelor|master)', df['requirements'][j].lower()).group()] )
            
            else:
                education_list.append(0)


        df['education_level'] = education_list

        return df



    def seniority(self, df):

        seniority_list=[]
        for j in range(len(df)):

            if re.findall(r'senior', df['requirements'][j].lower()):
                seniority_list.append(2)
            
            elif re.findall(r'junior', df['requirements'][j].lower()):
                seniority_list.append(1)

            else:
                seniority_list.append(0)

        df['job_title'] = seniority_list

        return df


    def get_states(self, df):

        with open('utils/states.txt', 'r') as f:
            states = f.read()
            states_list = states.split(',')
        f.close()
        
        job_states = []

        for j in range(len(df)):
            counter = 0
            
            for i in states_list:
                if re.findall(i, df['Location'][j].lower()):
                    job_states.append(states_list.index(i))
                    counter = 1
                    break
            
            if counter == 0:
                job_states.append(states_list.index('State_missing'))

        df['State'] = job_states
        
        return df




    def city(self, df):

        with open('utils/cities.txt', 'r') as f:
            cities = f.read()
            cities_list = cities.split(',')
        f.close()

        job_cities = []

        for j in range(len(df)):
            counter = 0
            
            for i in cities_list:
                if re.findall(i, df['Location'][j].lower()):
                    job_cities.append(cities_list.index(i))
                    counter = 1
                    break
            
            if counter == 0:
                job_cities.append(cities_list.index('city_missing'))
                
        df['Cities'] = job_cities
        
        return df
    

    def final_operations(self, df):
    
        # remove columns with constant values
        df['Company'] = df['Company'].apply(lambda x: self.preprocess_text(str(x)))
        df['Job_position'] = df['Job_position'].apply(lambda x: self.preprocess_text(str(x)))
        df['requirements'] = df['requirements'].fillna('')
        df['requirements'] = df['requirements'].apply(lambda x: self.preprocess_text(str(x)))
        df['job_descr_len'] = df['requirements'].apply(lambda x: 0 if not x else len(x))
        df.drop(['experience', 'Location', 'link', 'posting_time'], axis=1, inplace=True)

        df = df.loc[:, (df != df.iloc[0]).any()] 

        return df
    
    def __call__(self, df):
        df = self.clean_data(df)
        df = self.calc_salary(df)
        df = self.work_location(df)
        df = self.calc_experience(df)
        df = self.education(df)
        df = self.seniority(df)
        df = self.get_states(df)
        df = self.city(df)
        df = self.final_operations(df)
        return df
    
    
    
    

class PreprocessNew:

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = ' '.join([word for word in text.split() if not word in set(stopwords.words('english'))])
        text = ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

        return text
    
    def clean_data(self, df):
        #Some company posted for the same profile many times after a gap of few days   
        # we can store this information in the column 'posting frequency'
        # calculate posting frequency on the basis of company
        freq = df['Company'].value_counts()

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['posting_frequency'] = df['Company'].map(freq)

        # those not repeated will be null, therefore fill them as 1
        df['posting_frequency'].fillna(1, inplace=True)

        # We just deleted duplicates but we still see multiple entries for some companies
        # It looks like recently posted jobs with new tag are causing this, 
        # lets remove them
        df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace('\nnew','').lower())


        df['rating'] = df.rating.apply(lambda x: str(x).replace('\n',''))
        df['rating'] = pd.to_numeric(df['rating'].replace({"na":np.nan}))


        """
         Rows with missing salaries contain valuable information regarding job position, location and their requirements
         So we will keep them for now 
         for now lets fill them with -999
        """

        df['Salary'].fillna('-999', inplace=True)

        # remove new line and ruppes symbol  
        df['Salary'] = df['Salary'].apply(lambda x: str(x).replace('\n','').replace('₹',''))

        return df



    # to calculate max and min Salary per annum
    def calc_salary(self, df):

        yearly_min = {}
        yearly_max = {}

        for i in range(0, len(df)):

            if df['Salary'][i] == '-999':
                yearly_min[i] = 0
                yearly_max[i] = 0

            if 'a year' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
                yearly_min[i] = int(sal_min)

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('a year','').replace(',','')
                    yearly_max[i] = int(sal_max)

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('a year','').replace(',','')
                    yearly_max[i] = int(sal_max)


            if 'a month' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
                yearly_min[i] = int(sal_min) * 12

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('a month','').replace(',','')
                    yearly_max[i] = int(sal_max) * 12    

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('a month','').replace(',','')
                    yearly_max[i] = int(sal_max) * 12


            if 'an hour' in df['Salary'][i]:
                sal_min = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','').replace(' ','')
                yearly_min[i] = float(sal_min) * 9 * 22 * 12

                try:
                    sal_max = df['Salary'][i].split('-')[1].replace('an hour','').replace(',','').replace(' ','')
                    yearly_max[i] = float(sal_max) * 9 * 22 * 12  

                # if only single value present will be stored in both max and min, so the average comes accuate
                except:
                    sal_max = df['Salary'][i].split('-')[0].replace('an hour','').replace(',','').replace(' ','')
                    yearly_max[i] = float(sal_max) * 9 * 22 * 12

        # min, max and avg Salary columns
        df['min_Salary'] = pd.DataFrame(yearly_min.values(), index= yearly_min.keys())
        df['max_Salary'] = pd.DataFrame(yearly_max.values(), index= yearly_max.keys())
        df['avg_yearly_sal'] = ( df['min_Salary'] + df['max_Salary'] )/2.

        #Lets just drop these as we got our target column
        df.drop(['max_Salary', 'min_Salary', 'Salary'], axis=1, inplace=True)

        df['avg_yearly_sal'].fillna(0, inplace=True)

        return df



    def work_location(self, df):

        work_type = []
        for j in range(len(df)):

            if re.findall('full.time?', df['requirements'][j].lower()):
                work_type.append(2)

            elif re.findall('remote?', df['requirements'][j].lower()):
                work_type.append(1)

            else:
                work_type.append(0)


        df['work_category'] = work_type

        return df



    def calc_experience(self, df):

        experience_list = []
        for j in range(len(df)):

            if re.findall(r'\d+ year', df['requirements'][j]):
                experience_list.append(re.search(r'\d+ year', df['requirements'][j]).group()[0])

            else:
                experience_list.append('')

        df['net_experience'] = experience_list
        df['net_experience'] = pd.to_numeric(df['net_experience'])

        return df



    #Educational criteria mentioned by these companies can also be useful
    def education(self, df):

        education_list = []
        education_dict = {'bachelor':1, 'master':2, 'graduate':3}
        for j in range(len(df)):
            
            if re.findall(r'(graduate|bachelor|master)', df['requirements'][j].lower()):
                education_list.append( education_dict[re.search(r'(graduate|bachelor|master)', df['requirements'][j].lower()).group()] )
            
            else:
                education_list.append(0)


        df['education_level'] = education_list

        return df



    def seniority(self, df):

        seniority_list=[]
        for j in range(len(df)):

            if re.findall(r'senior', df['requirements'][j].lower()):
                seniority_list.append(2)
            
            elif re.findall(r'junior', df['requirements'][j].lower()):
                seniority_list.append(1)

            else:
                seniority_list.append(0)

        df['job_title'] = seniority_list

        return df


    def get_states(self, df):

        with open('utils/states.txt', 'r') as f:
            states = f.read()
            states_list = states.split(',')
        f.close()

        job_states = []

        for j in range(len(df)):
            counter = 0

            for i in states_list:
                if re.findall(i, df['Company'][j].lower()):
                    job_states.append(states_list.index(i))
                    df.loc[j, 'Company'] = re.sub(i, '',  df['Company'][j].lower())
                    counter = 1
                    break
                
            if counter == 0:
                job_states.append(states_list.index('State_missing'))

        df['State'] = job_states

        return df


    def replace_comp(self, df):

        for j in range(len(df)):

            if re.findall(r'(ltd|review|reviews|remote|temporarily|\d.?)', df['Company'][j].lower()):
                df.loc[j, 'Company'] = re.sub(r'(ltd|review|reviews|remote|temporarily|\d.?)', '', df['Company'][j].lower())
            
        return df


    def city(self, df):

        with open('utils/cities.txt', 'r') as f:
            cities = f.read()
            cities_list = cities.split(',')
        f.close()

        job_cities = []

        for j in range(len(df)):
            counter = 0

            for i in cities_list:
                if re.findall(i, df['Company'][j].lower()):
                    job_cities.append(cities_list.index(i))
                    df.loc[j, 'Company'] = re.sub(i, ' ', df['Company'][j].lower())
                    counter = 1
                    break
                
            if counter == 0:
                job_cities.append(cities_list.index('city_missing'))

        df['Cities'] = job_cities

        return df
    

    def final_operations(self, df):
    
        # remove columns with constant values
        df['Company'] = df['Company'].apply(lambda x: self.preprocess_text(str(x)))
        df['Job_position'] = df['Job_position'].apply(lambda x: self.preprocess_text(str(x)))
        df['requirements'] = df['requirements'].fillna('')
        df['requirements'] = df['requirements'].apply(lambda x: self.preprocess_text(str(x)))
        df['job_descr_len'] = df['requirements'].apply(lambda x: 0 if not x else len(x))

        df = df.loc[:, (df != df.iloc[0]).any()] 

        return df
    
    def __call__(self, df):
        df = self.clean_data(df)
        df = self.calc_salary(df)
        df = self.work_location(df)
        df = self.calc_experience(df)
        df = self.education(df)
        df = self.seniority(df)
        df = self.get_states(df)
        df = self.replace_comp(df)
        df = self.city(df)
        df = self.final_operations(df)
        return df