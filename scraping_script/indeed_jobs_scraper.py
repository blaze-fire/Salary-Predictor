# a simple scraping script to scrape indeed jobs

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def find_jobs_from():    
    """
    This function extracts all the desired characteristics of all new job postings
    of the title and States specified and returns them in single file.
    The arguments it takes are:
        - Job_title
        - States
        - Salary
        - Job Description
        - link
        - Date
    
    Note : As would be difficult to scrape from next pages as the results in a single page are limited, alternatively search for all the states
           individually and then concatenate all the results. 
    """
    
    urls = [
            'https://in.indeed.com/jobs?q=software&fromage=last&start=',
            'https://in.indeed.com/jobs?q=developer&l=india&start='
            'https://in.indeed.com/jobs?q=machine+learning&l=india&start=',
            'https://in.indeed.com/jobs?q=analyst&l=india&start=',
            'https://in.indeed.com/jobs?q=IT&l=india&start=',
            ]
    
    for url in urls:
        print(url)
        for i in range(1,501,30):                                                                 # to get results from other pages
            url = url + str(i)
            job_soup = load_indeed_jobs_div(url)
            jobs_list, num_listings = extract_job_information_indeed(job_soup)
            print('{} new job postings retrieved saved      {}'.format(num_listings, i))            #to keep note of the page currently scraping
            print('\n')
            save_jobs_to_excel(jobs_list)
            time.sleep(3)                                                                           # make sure to delay b/w scraping as the site might identify you as a bot and will block you :)
        

## ======================= SAVE FILES IN CSV AND APPEND THEM AS SCRAPING ======================= ##

def save_jobs_to_excel(jobs_list):
    jobs = pd.DataFrame(jobs_list)
    jobs.to_csv('./data/kaggle_data.csv', mode = 'a', index = False, header = None, encoding='utf-8-sig')



## ================== FUNCTIONS FOR INDEED.CO.IN =================== ##

def load_indeed_jobs_div(url):    
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup


def extract_job_information_indeed(soup):
    cols = []
    extracted_info = []

    titles = []
    cols.append('titles')

    comp_and_loc = []
    cols.append('company_and_location')

    salary = []
    cols.append('salary')

    requirements = []
    cols.append('requirements')

    ratings = []
    cols.append('ratings')
    
    for a_tag in soup.select('a[href*="/pagead"]'):
        url = 'https://in.indeed.com' + a_tag['href']
        
        try:
            page_temp = requests.get(url)

            temp = BeautifulSoup(page_temp.content, "html.parser")

            temp_soup = temp.find(id="viewJobSSRRoot")

            titles.append(extract_title(temp_soup))
            
            comp_and_loc.append(company_and_location(temp_soup))
            
            salary.append(extract_salary(temp_soup))
            
            requirements.append(extract_JD(temp_soup))

            ratings.append(extract_rating(temp_soup))

            time.sleep(2)
        except:
            pass
    

    
    
    for a_tag in soup.select('a[href*="/company"]'):
        url = 'https://in.indeed.com' + a_tag['href']
        

        try:
            page_temp = requests.get(url)

            temp = BeautifulSoup(page_temp.content, "html.parser")

            temp_soup = temp.find(id="viewJobSSRRoot")

            titles.append(extract_title(temp_soup))
            
            comp_and_loc.append(company_and_location(temp_soup))
            
            salary.append(extract_salary(temp_soup))
            
            requirements.append(extract_JD(temp_soup))
            
            ratings.append(extract_rating(temp_soup))

            time.sleep(2)

        except:
            pass


    extracted_info.append(titles)                    
    extracted_info.append(comp_and_loc)
    extracted_info.append(salary)
    extracted_info.append(requirements)
    extracted_info.append(ratings)


    jobs_list = {}
    
    for j in range(len(cols)):
        jobs_list[cols[j]] = extracted_info[j]
    
    num_listings = len(extracted_info[0])
    
    return jobs_list, num_listings


def extract_title(tag):
    title = tag.find(class_ = 'jobsearch-JobInfoHeader-title-container').text

    return title

def company_and_location(tag):
    try:
        text = tag.find(class_ = 'jobsearch-CompanyInfoWithoutHeaderImage').text
        return text
    except:
        return 'na'

def extract_salary(tag):
    try:
        salary = tag.find(class_ = 'jobsearch-JobMetadataHeader-item').text
        return salary
    except:
        return 'na'


def extract_JD(tag):
    try:
        jd = tag.find(class_ = 'jobsearch-jobDescriptionText').text
        return jd
    except:
        return 'na'


def extract_rating(tag):
    try:
        rating = tag.find("meta")
        return rating["content"]
    
    except:
        return 'na'

if __name__ == '__main__':
    while True:
        find_jobs_from()

    
