#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:50:36 2022

@author: nicholasrenegar
"""

import os
import pandas as pd

os.chdir("**** github/dir")
from src import Scraping_PointsPredictions
from src import Scraping_Injuries
from src import DataStaging
from src import APICalls
from src import ExpertSearch
from src import MIPS
from src import EvaluateContestResults
from src import AggregateResults
from src import InternalPointsProjections

#Filepaths
path_to_proj = "******* data/folder/path"

### USER INPUTS
date_string_list = pd.read_csv(f"{path_to_proj}/Model/_Mappings/DateList.csv")['Date'].astype(str).tolist()
web_scraping = 0
create_lineups = 0
run_MIPS = 0
evaluate_contest_results = 1
aggregate_contest_results = 1

#MIPS Model Hyperparameters
points_system_list = ["points_LR", "points_XGB", "points_dff_fsp"]
num_lineups_list = [150]
num_overlap_list = [6, 7]
max_appearances_list = [150]
alpha_list = [0.0]

#################################################################################
#######     Web Scraping
#################################################################################
if web_scraping == 1:
    
    data_folder = os.path.join(path_to_proj, "Data")

    for date_string in date_string_list:
        #Download Points Predictions
        browser_pts=Scraping_PointsPredictions.setup_browser(data_folder)
        Scraping_PointsPredictions.download_pts_proj(browser_pts, data_folder, date_string)       
        #Download Injury Data
        browser_inj = Scraping_Injuries.setup_browser(data_folder)
        Scraping_Injuries.download_injuries(browser_inj, date_string, data_folder)


#################################################################################
#######     Data Staging and MIPS Input
#################################################################################
if create_lineups == 1: 
    
    #Create MIPS Input
    for date_string in date_string_list:
        print(f"Processing date: {date_string}")
        
        #Initial Data Combination DraftKings Salaries and Scraped Data
        player_salaries = DataStaging.combine_data(date_string, path_to_proj)
    
        #Add API Data and Expert Takes
        player_salaries = APICalls.api_calls(player_salaries)
        player_salaries = ExpertSearch.run_expert_analysis(date_string, path_to_proj, player_salaries)
        
        #Create Internal Points Prediction
        player_salaries = InternalPointsProjections.live_points_prediction(player_salaries, date_string, path_to_proj)
    
        #Final Data Staging for MIPS
        player_salaries = DataStaging.prepare_final_dataframe(date_string, path_to_proj, player_salaries)
        
        #Save the MIPS Input to CSV
        output_file_path = os.path.join(path_to_proj, f'Model/Output/Model Input/Model Input{date_string}.csv')        
        player_salaries.to_csv(output_file_path, index=False)


#################################################################################
#######     MIPS Code
#################################################################################
if run_MIPS == 1:

    # Run the function
    MIPS.run_julia_mips(date_string_list, points_system_list, num_lineups_list, num_overlap_list, max_appearances_list, alpha_list, path_to_proj)


#################################################################################
#######     Evaluate Contest Results
#################################################################################
if evaluate_contest_results == 1:
    
    #Evaluate contest for all date strings
    for date_string in date_string_list:
        print(f"Evaluating contest results - date: {date_string}")
        curr_contest_results = EvaluateContestResults.evaluate_draftkings_results(date_string, path_to_proj)
    
        #Add date_string to full historical list if it is absent
        EvaluateContestResults.update_date_list(date_string, path_to_proj)

#################################################################################
#######     Aggregate Contest Results
#################################################################################
if aggregate_contest_results == 1:
    
    #Aggregate All Results
    aggregate_results = AggregateResults.aggregate_and_plot_results(date_string, path_to_proj, num_lineups=150)
