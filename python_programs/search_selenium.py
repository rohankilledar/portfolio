from time import sleep
import pandas as pd
from selenium import webdriver


df = pd.read_excel(r'C:\Users\Rohan\Downloads\harry_import.xlsx')
PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(PATH)
url = 'https://www.google.com'

keywords = df['Harry'].values.tolist()

for keyword in keywords:
    driver.get(url)
    searchBar = driver.find_element_by_name('q')
    searchBar.send_keys(keyword)
    searchBar.send_keys('\n')
    sleep(2)