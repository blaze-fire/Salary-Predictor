# a simple scraping script to scrape indeed jobs

import urllib
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
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
    
    for i in range(1,50001,20):                                                                 # to get results from other pages 
        
        job_soup = load_indeed_jobs_div(i)
        jobs_list, num_listings = extract_job_information_indeed(job_soup)

        save_jobs_to_excel(jobs_list)
        print('{} new job postings retrieved saved      {}'.format(num_listings, i))            #to keep note of the page currently scraping
        print('\n')
        time.sleep(2)                                                                           # make sure to delay b/w scraping as the site might identify you as a bot and will block you :)
    

## ======================= SAVE FILES IN CSV AND APPEND THEM AS SCRAPING ======================= ##

def save_jobs_to_excel(jobs_list):
    jobs = pd.DataFrame(jobs_list)
    jobs.to_csv('C:/Users/krish/Music/Job_ML_project/data/raw_data.csv', mode = 'a', index = False, header = None, encoding='utf-8-sig')



## ================== FUNCTIONS FOR INDEED.CO.IN =================== ##

def load_indeed_jobs_div(i):
    url = ('https://in.indeed.com/jobs?q=software+developer&fromage=last&start=' + str(i))

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    job_soup = soup.find(id="resultsCol")
    return job_soup

def extract_job_information_indeed(job_soup):
    job_elems = job_soup.find_all('div', class_='jobsearch-SerpJobCard')
    locations = job_soup.find_all(class_='location accessible-contrast-color-location')
    cols = []
    extracted_info = []
    
    
    
    titles = []
    cols.append('titles')
    for job_elem in job_elems:
        titles.append(extract_job_title_indeed(job_elem))
    extracted_info.append(titles)                    

    companies = []
    cols.append('companies')
    for job_elem in job_elems:
        companies.append(extract_company_indeed(job_elem))
    extracted_info.append(companies)

    location = []
    cols.append('location')
    for x in locations:
        x = BeautifulSoup.get_text(x)
        location.append(x)
    extracted_info.append(location)

    salary = []
    cols.append('salary')
    for job_elem in job_elems:
        salary.append(extract_salary(job_elem))
    extracted_info.append(salary)

    dates = []
    cols.append('date_listed')
    for job_elem in job_elems:
        dates.append(extract_date_indeed(job_elem))
    extracted_info.append(dates)

    qualifications = []
    cols.append('qualifications')
    for job_elem in job_elems:
        qualifications.append(extract_qualifications(job_elem))
    extracted_info.append(qualifications)

    ratings = []
    cols.append('ratings')
    for job_elem in job_elems:
        ratings.append(extract_rating(job_elem))
    extracted_info.append(ratings)

    requirements = []
    cols.append('requirements')
    for job_elem in job_elems:
        requirements.append(extract_requirements(job_elem))
    extracted_info.append(requirements)

    links = []
    cols.append('links')
    for job_elem in job_elems:
        links.append(extract_link_indeed(job_elem))
    extracted_info.append(links)
    
    jobs_list = {}
    
    for j in range(len(cols)):
        jobs_list[cols[j]] = extracted_info[j]
    
    num_listings = len(extracted_info[0])
    
    return jobs_list, num_listings


def extract_job_title_indeed(job_elem):
    title_elem = job_elem.find('h2', class_='title')
    title = title_elem.text.strip()
    return title

def extract_company_indeed(job_elem):
    company_elem = job_elem.find('span', class_='company')
    company = company_elem.text.strip()
    return company

def extract_link_indeed(job_elem):
    link = job_elem.find('a')['href']
    link = 'https://in.indeed.com/' + link
    return link

def extract_date_indeed(job_elem):
    date_elem = job_elem.find('span', class_='date')
    date = date_elem.text.strip()
    return date

def extract_location(job_elem):
    locate = job_elem.find(class_='location accessible-contrast-color-location')
    if(locate):
        locate = BeautifulSoup.get_text(locate)
        return locate
    else:
        return 'na'

def extract_rating(job_elem):
    rating = job_elem.find('span', class_='ratingsContent')
    if(rating):
        rating = BeautifulSoup.get_text(rating)
        return rating
    else:
        return 'na'

def extract_requirements(job_elem):
    requirements = job_elem.find(class_='jobCardReqList')
    if(requirements):
        requirements = BeautifulSoup.get_text(requirements)
        return requirements
    else:
        return 'na'

def extract_qualifications(job_elem):
    qual_elem = job_elem.find(class_='summary')
    if(qual_elem):
        qualifications = BeautifulSoup.get_text(qual_elem)              # To deal with text present b/w html tags 
        qualifications = qualifications.replace('\n','')
        return qualifications
    else:
        return 'na'

def extract_salary(job_elem):
    salary_elm =  job_elem.find('span', class_='salaryText')
    if(salary_elm):
        salary = BeautifulSoup.get_text(salary_elm)                     # To deal with text present b/w html tags 
        return salary
    else:
        return 'na'


if __name__ == '__main__':
    while True:
        find_jobs_from()

    
