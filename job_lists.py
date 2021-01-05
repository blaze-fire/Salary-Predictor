# a simple scraping script to scrape indeed jobs

import urllib
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time



def load_list():
    url = ('https://en.wikipedia.org/wiki/List_of_programming_languages')

    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    job_soup = soup.find(id="bodyContent")
    jobs = pd.DataFrame(jobs_list)
    jobs.to_csv('C:/Users/krish/Desktop/Job_ML_project/fresher.csv', mode = 'a', index = False, header = None, encoding='utf-8-sig')

## ================== TO AUTOMATE THE PROCESS  =================== ##

if __name__ == '__main__':
    while True:
        load_list()

    
