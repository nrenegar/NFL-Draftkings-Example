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
    download_path = os.path.join(data_folder, "Points Projections")

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


######################################################
############Rotowire Points Projections###############
######################################################
def scrape_rotowire_pts(browser, data_folder, date_string):
    browser.get("https://www.rotowire.com/users/login.php?go=%2Fusers%2Flogin")
    
    #Login
    time.sleep(2)
    usernameLink =  browser.find_element("xpath",'/html/body/div[1]/div/main/div/div[1]/form/input[1]')
    usernameLink.click()
    usernameLink.send_keys("******************* username")
    
    time.sleep(2)
    usernameLink =  browser.find_element("xpath",'/html/body/div[1]/div/main/div/div[1]/form/input[2]')
    usernameLink.click()
    usernameLink.send_keys("******************** password")
    usernameLink.send_keys(Keys.RETURN)
    
    #Get Projections
    time.sleep(2)
    browser.get("https://www.rotowire.com/daily/nfl/value-report.php?site=DraftKings")
    time.sleep(5)
    nflDownloadLink =  browser.find_element("xpath",'//*[@id="root"]/div/div[2]/div[4]/div[3]/div[2]/button[2]/div')
    nflDownloadLink.click()
    
    #Rename file with today's date
    time.sleep(3)
    download_path = os.path.join(data_folder, "Points Projections")
    downloaded_file = os.path.join(download_path, 'rw-nfl-player-pool.csv')
    new_file_name = os.path.join(download_path, f'rotowire-NFL-players{date_string}.csv')
    
    # Rename file with today's date
    if os.path.exists(downloaded_file):
        os.rename(downloaded_file, new_file_name)
        print(f"File renamed to {new_file_name}")
    else:
        print(f"File {downloaded_file} not found.")
    

##################################################
############FSP Points Projections################
##################################################
def scrape_fsp_pts(browser, data_folder, date_string):
    browser.get("https://dfs-projections.wetalkfantasysports.com/shiny/DFS/#tab-8997-2")
    
    # Get the points projections.
    nflLink =  browser.find_element("xpath",'/html/body/nav/div/ul/li[1]/a')
    nflLink.click()
               
    nflDKLink =  browser.find_element("xpath",'/html/body/nav/div/ul/li[1]/ul/li[2]/a')
    nflDKLink.click()
    
    time.sleep(3)
    nflDownloadLink  =  browser.find_element("xpath",'//*[@id="downloadData6"]')
    nflDownloadLink.click()

    #Rename file with today's date
    time.sleep(3)
    download_path = os.path.join(data_folder, "Points Projections")
    downloaded_file = os.path.join(download_path, 'NFLDK.csv')
    new_file_name = os.path.join(download_path, f'NFLDK{date_string}.csv')
    
    # Rename file with today's date
    if os.path.exists(downloaded_file):
        os.rename(downloaded_file, new_file_name)
        print(f"File renamed to {new_file_name}")
    else:
        print(f"File {downloaded_file} not found.")
    


##################################################
############DFF Points Projections################
##################################################
def scrape_dff_pts(browser, date_string):
    browser.get("https://www.dailyfantasyfuel.com/nfl/projections/draftkings")
    
    
    ProjectionsLink =  browser.find_element(By.CLASS_NAME, 'projections-download')
    ProjectionsLink.click()

##################################################
############Run all Downloads################
##################################################

def download_pts_proj(browser, data_folder, date_string):
    scrape_rotowire_pts(browser, data_folder, date_string)
    scrape_fsp_pts(browser, data_folder, date_string)
    scrape_dff_pts(browser, data_folder)
    browser.close()
