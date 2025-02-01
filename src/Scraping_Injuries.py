# coding: utf-8

import pandas as pd
import datetime
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import requests
from bs4 import BeautifulSoup


#Setup browser with right download folder
def setup_browser(data_folder):
    # Set the download directory to be inside the data_folder
    download_path = os.path.join(data_folder, "Injuries")

    chrome_options = webdriver.ChromeOptions()
    prefs = {
        'download.default_directory': str(download_path),
        "savefile.default_directory": str(download_path),
        "download.prompt_for_download": False,  # To auto-download the file
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option('prefs', prefs)

    path_to_chromedriver = '/Applications/chromedriver'
    browser = webdriver.Chrome(options=chrome_options, service=Service(path_to_chromedriver))
    

    return browser


##################################################
########NFL injury report###################
##################################################
def scrape_official_inuries(browser, date_string, data_folder):
    # Set the download directory to be inside the data_folder
    download_path = os.path.join(data_folder, "Injuries")

    url ="https://www.nfl.com/injuries/"
    browser.get(url)
    time.sleep(2)

    
    # Find all player names using the updated HTML structure
    player_elements = browser.find_elements(By.CSS_SELECTOR, 'tbody tr td a.nfl-o-cta--link')

    player_names = []

    for player_element in player_elements:
        name = player_element.get_attribute("innerText")
        player_names.append(name)

    # Create a DataFrame from the list of player names
    df = pd.DataFrame(player_names, columns=["Name"])

    df.to_csv(os.path.join(download_path, f'Injuries_Rotoworld{date_string}.csv'), index=False)

##################################################
########CBS Sports injury report##################
##################################################
def scrape_cbs_injuries(browser, date_string, data_folder):
    # Set the download directory to be inside the data_folder
    download_path = os.path.join(data_folder, "Injuries")

    url ='https://www.cbssports.com/nfl/injuries/daily'
    browser.get(url)
    time.sleep(2)
    browser.execute_script("window.stop();")

    # Get all names from injury report
    elements = browser.find_elements(By.CLASS_NAME, 'CellPlayerName--long')
    
    names_list = []  # Create an empty list to hold the names
    
    # Loop through and get text from all elements
    for element in elements:
        name = element.get_attribute("innerText")
        names_list.append(name)  # Append the name to the list
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(names_list, columns=["Name"])
    
    df.columns = ["Name"]
    df.to_csv(str(download_path+'/Injuries_CBS'+date_string+'.csv'),index=False)
    browser.close()

##################################################
############Run all Downloads################
##################################################

def download_injuries(browser, date_string, data_folder):
    scrape_official_inuries(browser, date_string, data_folder)
    scrape_cbs_injuries(browser, date_string, data_folder)