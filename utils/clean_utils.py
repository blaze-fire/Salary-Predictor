import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

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





# to calculate max and min Salary per annum
def calc_salary(df):
    
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
    df['avg_yearly_sal'] = ( df['min_Salary'] + df['max_Salary'] )/2
    df['monthly_Salary'] = df['avg_yearly_sal']/12.

    df['avg_yearly_sal'].fillna(0, inplace=True)
    df['min_Salary'].fillna(0, inplace=True)
    df['max_Salary'].fillna(0, inplace=True)
    df['monthly_Salary'].fillna(0, inplace=True)

    #Lets just drop these as we got our target column
    df.drop('max_Salary', axis=1, inplace=True)
    df.drop('min_Salary', axis=1, inplace=True)
    df.drop('monthly_Salary', axis=1, inplace=True)

    #we can divide the annual Salary into 6 differrent categories
    df['income_cat'] = pd.cut(df['avg_yearly_sal'], bins=[-999, 0, 50000, 100000, 500000, 1000000, 2500000, np.inf], labels=[-1, 1, 2, 3, 4, 5, 6])
    df = df.drop('Salary', axis=1)

    return df
                




def calc_experience(df):
    #Experience is mentioned in both requirements and experience so we will collect them all and save it in a column of experience
    #Some of these requirements mention experienced
    df['experience'] = df['experience'].fillna('na')

    net_experience = []
    for i in df.experience:
        temp=[]
        for word in i.split():
            if word.isdigit():
                temp.append(word)
        if temp:
            temp.sort(reverse=True)
            net_experience.append(temp[0])
        else:
            net_experience.append(-99)

    df['net_experience'] = net_experience

    df['net_experience'] = df['net_experience'].astype('int32')

    def clean(x):
        for p in ['â', '€', '¦', '“', '¢', '™']:
            x.replace(p, ' ')
        return x

    df['requirements'] = df['requirements'].map(clean)
    
    net_experience = []
    for i in df.requirements:
        temp=[]
        for word in i.split():
            if word.isdigit():
                temp.append(word)
        if temp:
            temp.sort(reverse=True)
            net_experience.append(temp[0])
        else:
            net_experience.append(-99)
    
    df['exp2'] = net_experience

    #Removing unwanted values from experience column
    for p in ['²', '0080091', '2020', '2024', '2019', '90', '88', '32', '48', '40', '50', '24']:
        df['exp2'] = df['exp2'].apply(lambda x: str(x).replace(p,'-99'))

    df['exp2'] = df['exp2'].apply(lambda x: int(x) if x.isdigit() else -99)

    #where experience required is mentioned in requirements column but missing in experience column
    df['net_experience'] = df['net_experience'].where((df['net_experience']>0), df['exp2'])
    df.drop('exp2', axis=1, inplace=True)

    df.loc[294, 'experience']

    #Upon Observation some openings require no experience
    df.loc[[294, 14, 111, 122, 362, 749], 'net_experience'] = 0
    
    #Some job positions also mention titles like junior, intern etc. which require 0 experience, we also want to count that where net experience is missing
    for i in df.index:
        if df.loc[i, 'net_experience'] < 0:
            for word in str(df.Job_position[i]).lower().split():
                if word == 'jr' or word == 'junior' or word == 'fresher' or word == 'intern' or word == 'intership' or word == 'interns' or word == 'freshers':
                    df.loc[i, 'net_experience'] = 0
                else:
                    df.loc[i, 'net_experience'] = -99 


    return df





#Educational criteria mentioned by these companies can also be useful
def education(df):
    def education_level(data):
        if 'bachelor' in data.replace('year',' ').replace("'",' ').lower().split():
            return 'bachelor'
        if 'secondary' in data.replace('year',' ').replace('(',' ').replace("'",' ').lower().split():
            return 'secondary'
        if 'master' in data.replace('year',' ').replace("'",' ').lower().split():
            return 'masters'

    df['education_level'] = df['experience'].map(education_level)
    df['education_level'].fillna('na',inplace=True)

    #As the categories of seniority is only jr, senior or na, we can one hot encode them
    df = pd.concat([df, pd.get_dummies(df['education_level'])], axis=1)

    return df


def senior(df):
    def seniority(title):
        title = str(title) 
        if 'ii' in title.lower().split() or 'director' in title.lower().split() or 'specialist' in title.lower().split() or 'professional' in title.lower().split() or 'sr.' in title.lower().split() or 'senior' in title.lower().split():
            return 'senior'
        elif 'i' in title.lower().split() or 'associate' in title.lower().split() or 'junior' in title.lower().split() or 'jr' in title.lower().split()  or 'jr.' in title.lower().split() or 'trainee' in title.lower().split() or 'intern' in title.lower().split() or 'jr.' in title.lower().split():
            return 'jr'
        else:
            return 'na'
    # to calculate the seniority of the position applying for
    df['job_title'] = df['Job_position'].apply(seniority)

    '''
        For encoding rank transforamtion, label encoding, frequency encoding were applied but they had very weak correlation with avg_year_Salary
        as the categories of seniority is only jr, senior or na, we can one hot encode them
    '''

    df = pd.concat([df, pd.get_dummies(df['job_title'])], axis=1)

    return df




'''
    Upon analyzing the requirements column following are the most popular professions
    lets store their frequencies
'''

def profession(df):
    def calc_jobs(data):
        data = data.lower().replace(' ', '')
        if 'machinelearning' in data:
            return 'machine learning'

        if 'datascientist' in data:
            return 'data scientist'

        if 'softwaredeveloper' in data:
            return 'software developer'

        if 'softwareengineer' in data:
            return 'software engineer'

        if 'deeplearning' in data:
            return 'deep learning'

    df['popular_profession'] = df['requirements'].apply(calc_jobs)
    df['popular_profession'] = df['popular_profession'].fillna('na')
    df = pd.concat([df, pd.get_dummies(df['popular_profession'])], axis=1)
    
    return df





def one_hot_encode_cols(df):
    # we can split the location column and get the state 
    df['State'] = df['Location'].apply(lambda x:  x.split(',')[1] if len(x.split(',')) > 1 else x)

    #We can one hot encode these States values
    df = pd.concat([df, pd.get_dummies(df['State'])], axis=1)
    df['requirements'] = df['requirements'].fillna('')
    df['job_descr_len'] = df['requirements'].apply(lambda x: 0 if not x else len(x))

    return df



# Some companies have multiple job openings this could be useful
def job_openings(df):
    job_openings = df['Company'].value_counts()
    df['job_openings'] = df['Company'].map(job_openings)
    return df


'''
    Analyzing Job skills
        As due to covid-19 many people working in the industry have lost their jobs, and according to news articles the
        skill demand for job industry is also changing, lets take a look at the skills, in demand in the job industry
'''

def analyze_skills(df):
    requirements = df['requirements']
    requirements = list(filter(None, requirements))

    # split punctuation     
    for p in ['-','(',')','.','/']:
        job_descr = []
        for i in range(0, len(requirements)):
            c = requirements[i].split(p)
            for x in c:
                x.replace('.',' ')
                job_descr.append(x)


    # Remove punctuation and convert to lower case
    for x in range(0,len(job_descr)):
        for p in ['.', '-', ')', '(', '…', ',', ':', "'"]:
            job_descr[x] = job_descr[x].replace(p,' ')
        job_descr[x] = job_descr[x].lower()


    # analyzing keywords from custom keyword list
    f = open("./utils/skills.txt","r",) 
    skills=[]
    for x in f:
        skills.append(x)
    f.close()       
    
    for i in skills:
        skills = i.split(',')

    for i in range(0, len(skills)):
        skills[i] = skills[i].replace(' ','')
        skills[i] = skills[i].lower()

    df['requirements'] = df['requirements'].apply(lambda x: ' '.join([word for word in x.lower().split() if word in (skills)]))
    df['requirements'].replace(to_replace='', value='na', inplace=True)

    processed_text = word_tokenize(str(job_descr))

    # to calculate the frequency of a particular skill mentioned in job description
    def calc_skill_freq(data):
        skill_dict = {}

        for i in range(1,len(data)):
            token = data[i]
            if token in skills:
                try:
                    skill_dict[token].add(i)
                except:
                    skill_dict[token] = {i}

        for i in skill_dict:
            skill_dict[i] = len(skill_dict[i])

        return skill_dict
    
    job_descr_dict = calc_skill_freq(processed_text)

    '''
        Some Companies have mentioned the required skills in job position and some in some description
        Lets take a look at the skills mentioned in Job description column, then we will add them to get skills in demand
    '''

    # remove punctuation present in job position column
    def remove_punctuation(df):
        for p in ['/', ',', '(', ')', '-', '|', '&', '_', '.', '“', '”', ':']:
            df['Job_position'] = df['Job_position'].apply(lambda x: str(x).replace(p,' '))

        return df
    
    df = remove_punctuation(df)

    # analyzing stopwords from custom stopwords list
    f = open("./utils/stopwords.txt","r",) 
    stopwords=[]

    for x in f:
        stopwords.append(x)

    f.close()

    for i in stopwords:
        stopwords = i.split(',')
    
    for i in range(0, len(stopwords)):
        stopwords[i] = stopwords[i].replace("'","")
        stopwords[i] = stopwords[i].replace(" ","")
        stopwords[i] = stopwords[i].lower()
    
    # removing stopwords from the Job_position column
    job_role = list(df['Job_position'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (stopwords)])))

    df['Job_position'] = df['Job_position'].apply(lambda x: ' '.join([word for word in x.lower().split() if word in (skills)]))    
    df['Job_position'] = df['Job_position'].where(df['Job_position'] != '', 'na')    

    job_role = word_tokenize(str(job_role))

    # Now calculate the frequency of a particular skill mentioned in job role    
    job_role_dict = calc_skill_freq(job_role)

    '''
        Below we first pass all the elements of the first dictionary into the third one and then pass the second dictionary into the third. 
        This will replace the duplicate keys of the first dictionary.
        More info : (https://www.geeksforgeeks.org/python-merging-two-dictionaries/)
    '''

    skills_dict = {**job_role_dict, **job_descr_dict}

    #Lets save this dictionary for now it will be useful for EDA
    '''
        import pickle
        skill_file = open('./utils/skill_dictionary', 'wb') 
        pickle.dump(skills_dict, skill_file) 
        skill_file.close()
    '''


    # now create new column for each skill with value equal to the frequency of that skill occurring in that particular cell
    def calc_freq(df):
        for key in list(skills_dict.keys()):
            if skills_dict[key] > 15:
                skill_calc = []
                for i in range(0,len(df)):
                    count = 0
                    
                    # here we are counting frquency from both requirements and Job position column
                    for word in df['requirements'][i].lower().split() :
                        if key in df['Job_position'][i].lower().split():
                            count += 1
                        if key == word:
                            count += 1
                            skill_calc.append(count)
                        else:
                            skill_calc.append(0)
                            
                df = pd.concat([df, pd.DataFrame(skill_calc, columns=[key])], axis=1)
                # all the missing values should be filled with zero as they dont contain that particular skill 
                df[key] = df[key].fillna(0)
        return df
    df = calc_freq(df)
    
    '''
    As skills from job position and description were added its possible some of them dont appear in description, 
    their frequency wiil be zero so we must drop them
    '''

    # remove columns with constant values
    df = df.loc[:, (df != df.iloc[0]).any()] 

    return df



            
