#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:56:44 2022

@author: nicholasrenegar
"""

import os
import pandas as pd
import re
from datetime import datetime
from pytz import timezone
import numpy as np
from src import NameMatching

def load_mapping_name(path_to_proj):
    mapping_name_path = os.path.join(path_to_proj, "Model/_Mappings/mappingName.csv")
    mapping_name = pd.read_csv(mapping_name_path)
    mapping_name['Name'] = mapping_name['Name'].astype(str)
    mapping_name['Final'] = mapping_name['Final'].astype(str)
    return mapping_name

def load_salaries(path_to_proj, date_string):
    path_to_salaries = os.path.join(path_to_proj, f"Data/Salaries/DKSalaries{date_string}.csv")
    player_salaries = pd.read_csv(path_to_salaries).sort_values(by='Name')
    player_salaries['Name'] = player_salaries['Name'].astype(str)
    return player_salaries

def update_player_names(player_salaries, path_to_proj):
    mapping_name = load_mapping_name(path_to_proj)
    
    # Loop through each player name in player_salaries
    for name in player_salaries['Name']:
        if name not in mapping_name['Final'].values:
            # Remove any rows where mapping_name['Name'] == name
            mapping_name = mapping_name[mapping_name['Name'] != name]
            
            # Add a new row to mapping_name with the new Name and Final
            new_row = pd.DataFrame({'Name': [name], 'Final': [name]})
            mapping_name = pd.concat([mapping_name, new_row], ignore_index=True)
    
    # Save updated mapping_names to CSV
    mapping_name.to_csv(f"{path_to_proj}/Model/_Mappings/mappingName.csv", index=False)

    return mapping_name

##################################################
########NFL injury report###################
##################################################

def read_injury_data_cbs(path_to_proj, date_string, mapping_name, player_salaries):
    path_to_injuries_cbs = os.path.join(path_to_proj, f"Data/Injuries/Injuries_CBS{date_string}.csv")
    injuries_cbs = pd.read_csv(path_to_injuries_cbs).sort_values(by='Name')

    # Print and log all players we need to add to name mapping
    missing_names = injuries_cbs[~injuries_cbs['Name'].isin(mapping_name['Name'])]
    if not missing_names.empty:
        print(missing_names)

    # Map starter names to DraftKings names
    injuries_cbs['Name'] = injuries_cbs['Name'].apply(lambda x: mapping_name['Final'][mapping_name['Name'] == x].values[0] if x in mapping_name['Name'].values else x)

    # Flag for Injured Players
    player_salaries['Questionable'] = player_salaries['Name'].apply(lambda x: 1 if x in injuries_cbs['Name'].values else 0)

    return player_salaries

def read_injury_data_rotoworld(path_to_proj, date_string, mapping_name, player_salaries):
    path_to_injuries_rotoworld = os.path.join(path_to_proj, f"Data/Injuries/Injuries_Rotoworld{date_string}.csv")
    injuries_rotoworld = pd.read_csv(path_to_injuries_rotoworld).sort_values(by='Name')

    # Print and log all players we need to add to name mapping
    missing_names = injuries_rotoworld[~injuries_rotoworld['Name'].isin(mapping_name['Name'])]
    if not missing_names.empty:
        print(missing_names)

    # Map starter names to DraftKings names
    injuries_rotoworld['Name'] = injuries_rotoworld['Name'].apply(lambda x: mapping_name['Final'][mapping_name['Name'] == x].values[0] if x in mapping_name['Name'].values else x)

    # Remove Injured Players
    player_salaries['Questionable'] = player_salaries.apply(
        lambda row: 1 if row['Name'] in injuries_rotoworld['Name'].values else row['Questionable'], axis=1
    )

    return player_salaries

def adjust_questionable_status(player_salaries):
    # Adjust the questionable flag to 0 if the player's combined projections are 0
    player_salaries['Questionable'] = player_salaries.apply(
        lambda row: 0 if row['ppg_projection_DFF'] + row['ppg_projection_FSP'] + row['ppg_projection_RW'] == 0 else row['Questionable'],
        axis=1
    )
    
    # Adjust the questionable flag to 0 if the player is marked as "Out"
    player_salaries['Questionable'] = player_salaries.apply(
        lambda row: 0 if row['injury_status'] == "O" else row['Questionable'],
        axis=1
    )
    
    return player_salaries

##################################################
########External Points Predictions###################
##################################################

def add_points_predictions_dff(path_to_proj, date_string, player_salaries):
    mapping_name = load_mapping_name(path_to_proj)

    # Read in the DailyFantasyFuel points predictions
    date_string_full = f"{date_string[:4]}-{date_string[4:6]}-{date_string[6:]}"
    path_to_points_pred = os.path.join(path_to_proj, f'Data/Points Projections/DFF_NFL_cheatsheet_{date_string_full}.csv')
    points_pred = pd.read_csv(path_to_points_pred)
    
    # Rename 'L5_fppg_avg' to 'L5_ppg_avg' if it exists
    if 'L5_fppg_avg' in points_pred.columns:
        points_pred.rename(columns={'L5_fppg_avg': 'L5_ppg_avg'}, inplace=True)

    # Create a full name field for matching
    points_pred['First..Last'] = points_pred['first_name'] + " " + points_pred['last_name']
    points_pred['ppg_projection_DFF'] = points_pred['ppg_projection']
    
    # Replace null values in 'First..Last' with 'first_name'
    points_pred['First..Last'] = points_pred['First..Last'].fillna(points_pred['first_name'])

    # Log any players missing from the name mapping
    missing_names = points_pred[~points_pred['First..Last'].isin(mapping_name['Name'])]
    if not missing_names.empty:
        print(missing_names)
    
    # Map projections names to DraftKings names
    points_pred['Name'] = NameMatching.remap_player_names(points_pred['First..Last'], path_to_proj)
    
    # Merge the points predictions with the existing player salaries
    player_salaries_pred = player_salaries.merge(
        points_pred[['Name', 'ppg_projection_DFF', 'L5_ppg_avg', 'injury_status']],
        on='Name',
        how='left'
    )
    
    # Fill missing values with zeros or appropriate defaults
    player_salaries_pred['ppg_projection_DFF'].fillna(0, inplace=True)
    player_salaries_pred['L5_ppg_avg'].fillna(0, inplace=True)
    player_salaries_pred['injury_status'].fillna('', inplace=True)
    
    # Update the questionable status based on injury status
    player_salaries_pred['Questionable'] = player_salaries_pred.apply(
        lambda row: 1 if row['injury_status'] == "Q" else row['Questionable'], axis=1
    )
    
    # Process Vegas odds information
    vegas_data = points_pred.groupby(['team', 'spread', 'implied_team_score']).agg(total_salary=('salary', 'sum')).reset_index()
    vegas_data = vegas_data.sort_values('total_salary', ascending=False).drop_duplicates('team').drop(columns=['total_salary'])
    
    player_salaries_pred = player_salaries_pred.merge(vegas_data, left_on='TeamAbbrev', right_on='team', how='left').drop(columns=['team'])
    
    # Process defensive matchup information
    defensive_data = points_pred.groupby(['position', 'team', 'L5_dvp_rank']).agg(total_salary=('salary', 'sum')).reset_index()
    defensive_data = defensive_data.sort_values('total_salary', ascending=False).drop_duplicates(['position', 'team']).drop(columns=['total_salary'])
    
    player_salaries_pred = player_salaries_pred.merge(defensive_data, left_on=['Position', 'TeamAbbrev'], right_on=['position', 'team'], how='left').drop(columns=['team', 'position'])
    
    return player_salaries_pred



def add_points_predictions_fsp(path_to_proj, date_string, player_salaries):
    mapping_name = load_mapping_name(path_to_proj)

    # Read in the FantasySixPack points predictions
    path_to_points_pred_fsp = os.path.join(path_to_proj, f'Data/Points Projections/NFLDK{date_string}.csv')
    points_pred_fsp = pd.read_csv(path_to_points_pred_fsp)
    
    # Convert Player column to string for consistency
    points_pred_fsp['Player'] = points_pred_fsp['Player'].astype(str)
    
    # Log any players missing from the name mapping
    missing_names_fsp = points_pred_fsp[~points_pred_fsp['Player'].isin(mapping_name['Name'])]
    if not missing_names_fsp.empty:
        print(missing_names_fsp)
    
    # Map projections names to DraftKings names
    points_pred_fsp['Name'] = NameMatching.remap_player_names(points_pred_fsp['Player'], path_to_proj)
    
    # Add FantasySixPack points predictions to player salaries
    if 'Proj' in points_pred_fsp.columns:
        points_pred_fsp['ppg_projection_FSP'] = points_pred_fsp['Proj']
    elif 'Projection' in points_pred_fsp.columns:
        points_pred_fsp['ppg_projection_FSP'] = points_pred_fsp['Projection']
    else:
        raise KeyError("Neither 'Proj' nor 'Projection' column found in the DataFrame.")
    player_salaries_pred = player_salaries.merge(
        points_pred_fsp[['Name', 'ppg_projection_FSP']],
        on='Name',
        how='left'
    )
    
    # Fill missing values for FantasySixPack projections with 0
    player_salaries_pred['ppg_projection_FSP'].fillna(0, inplace=True)
    
    return player_salaries_pred

def add_points_predictions_rw(path_to_proj, date_string, player_salaries):
    mapping_name = load_mapping_name(path_to_proj)

    # Read in the Rotowire points predictions
    path_to_points_pred_rw = os.path.join(path_to_proj, f'Data/Points Projections/rotowire-NFL-players{date_string}.csv')
    points_pred_rw = pd.read_csv(path_to_points_pred_rw)
    
    # Convert PLAYER column to string for consistency
    points_pred_rw['PLAYER'] = points_pred_rw['PLAYER'].astype(str)
    
    # Log any players missing from the name mapping
    missing_names_rw = points_pred_rw[~points_pred_rw['PLAYER'].isin(mapping_name['Name'])]
    if not missing_names_rw.empty:
        print(missing_names_rw)
    
    # Map projections names to DraftKings names
    points_pred_rw['Name'] = NameMatching.remap_player_names(points_pred_rw['PLAYER'], path_to_proj)
    
    # Add Rotowire points predictions to player salaries
    points_pred_rw['ppg_projection_RW'] = points_pred_rw['FPTS']
    player_salaries_pred = player_salaries.merge(
        points_pred_rw[['Name', 'ppg_projection_RW']],
        on='Name',
        how='left'
    )
    
    # Fill missing values for Rotowire projections with 0
    player_salaries_pred['ppg_projection_RW'].fillna(0, inplace=True)
    
    # If needed, you can also calculate standard deviation and add it
    # player_salaries_pred['std_dev_RW'] = points_pred_rw.apply(
    #     lambda row: min(row['ppg_projection_RW'], (row['CEIL'] - row['ppg_projection_RW'])**0.5) if not pd.isna(row['CEIL']) else 0,
    #     axis=1
    # )
    # player_salaries_pred['std_dev_RW'].fillna(0, inplace=True)
    
    return player_salaries_pred


##################################################
########Stadium/Game Information###################
##################################################
def add_stadium_game_info(player_salaries):
    # Define stadium information
    stadium_info = pd.DataFrame({
        'Stadium': ['OAK','ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'],
        'Latitude': [37.456, 33.5277, 33.7554, 39.2780, 42.7738, 35.2258, 41.8623, 39.0955, 41.5061, 32.7473, 39.7439, 42.3400, 44.5013, 29.6847, 39.7601, 30.3240, 39.0490, 33.9536, 33.9536, 36.0908, 25.9580, 44.9740, 42.0909, 29.9511, 40.8135, 40.8135, 39.9008, 40.4468, 47.5952, 37.4032, 27.9763, 36.1665, 38.9076],
        'Longitude': [-122.122, -112.2626, -84.4010, -76.6227, -78.7870, -80.8531, -87.6167, -84.5161, -81.6995, -97.0945, -105.0201, -83.0456, -88.0622, -95.4109, -86.1639, -81.6373, -94.4839, -118.3387, -118.3387, -115.1835, -80.2389, -93.2577, -71.2643, -90.0812, -74.0745, -74.0745, -75.1675, -80.0158, -122.3316, -121.9699, -82.5033, -86.7713, -76.8645],
        'Is_Dome': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    })
    
    # Initialize lists for game teams and datetime
    game_teams = []
    game_datetime = []

    # Process each Game.Info entry
    for game_info in player_salaries['Game Info']:
        # Extract the home team (team after '@')
        team_match = re.search(r"@([A-Z]+)", game_info)
        game_teams.append(team_match.group(1) if team_match else "")
        
        # Extract and format the datetime
        datetime_match = re.search(r"(\d+/\d+/\d+ \d+:\d+PM ET)", game_info)
        if datetime_match:
            raw_datetime = datetime_match.group(1)
            # Remove the 'ET' and parse the datetime
            raw_datetime = raw_datetime.replace(" ET", "")
            formatted_datetime = datetime.strptime(raw_datetime, "%m/%d/%Y %I:%M%p").astimezone(timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S")
            game_datetime.append(formatted_datetime)
        else:
            game_datetime.append("")

    # Add game datetime and stadium information to DataFrame
    player_salaries['Game_DateTime'] = game_datetime
    player_salaries['Stadium'] = game_teams
    
    # Add indicator for home team
    player_salaries['HomeTeam'] = player_salaries.apply(lambda row: 1 if row['TeamAbbrev'] == row['Stadium'] else 0, axis=1)
    
    # Merge with stadium information
    player_salaries = player_salaries.merge(stadium_info, on='Stadium', how='left')

    # Rename "Game Info" to "GameInfo"
    player_salaries.rename(columns={"Game Info": "GameInfo"}, inplace=True)

    return player_salaries

##################################################
########Main Data Staging Functions###################
##################################################

def combine_data(date_string, path_to_proj):
    # Processing player salaries
    player_salaries = load_salaries(path_to_proj, date_string)
    mapping_name = update_player_names(player_salaries, path_to_proj)
    
    # Process injury data
    player_salaries = read_injury_data_cbs(path_to_proj, date_string, mapping_name, player_salaries)
    player_salaries = read_injury_data_rotoworld(path_to_proj, date_string, mapping_name, player_salaries)

    # Add points projections
    player_salaries = add_points_predictions_dff(path_to_proj, date_string, player_salaries)
    player_salaries = add_points_predictions_fsp(path_to_proj, date_string, player_salaries)
    player_salaries = add_points_predictions_rw(path_to_proj, date_string, player_salaries)
    
    # Adjust questionable status from injury data plus points projections
    player_salaries = adjust_questionable_status(player_salaries)
    
    #Add stadium and game information
    player_salaries = add_stadium_game_info(player_salaries)
    
    # Example of saving the result
    output_path = os.path.join(path_to_proj, f"Model/Output/Points Prediction/Points Prediction{date_string}.csv")
    player_salaries.to_csv(output_path, index=False)

    return player_salaries


def prepare_final_dataframe(date_string, path_to_proj, player_salaries):
    """
    Prepares the final dataframe for MIPS by adding necessary columns to the player_salaries dataframe.

    Args:
    date_string (str): Date string for which to prepare the final dataframe.
    path_to_proj (str): Base path to the project directory.

    Returns:
    player_salaries (DataFrame): The updated DataFrame with additional columns.
    """

    # Add necessary columns
    player_salaries['QB'] = (player_salaries['Position'] == 'QB').astype(int)
    player_salaries['RB'] = (player_salaries['Position'] == 'RB').astype(int)
    player_salaries['WR'] = (player_salaries['Position'] == 'WR').astype(int)
    player_salaries['TE'] = (player_salaries['Position'] == 'TE').astype(int)
    player_salaries['FLEX'] = player_salaries['Position'].isin(['RB', 'WR', 'TE']).astype(int)
    player_salaries['DST'] = (player_salaries['Position'] == 'DST').astype(int)

    # Add points predictions
    player_salaries['points_dff_fsp'] = np.where(
        player_salaries['Position'] == "DST",
        np.maximum(player_salaries['ppg_projection_DFF'], player_salaries['ppg_projection_FSP']),
        (player_salaries['ppg_projection_DFF'] + player_salaries['ppg_projection_FSP']) / 2
    )

    player_salaries['points_dff_fsp_rw'] = np.where(
        player_salaries['Position'] == "DST",
        np.maximum(player_salaries['ppg_projection_DFF'], player_salaries['ppg_projection_FSP'], player_salaries['ppg_projection_RW']),
        (player_salaries['ppg_projection_DFF'] + player_salaries['ppg_projection_FSP'] + player_salaries['ppg_projection_RW']) / 3
    )

    player_salaries['points_rw'] = player_salaries['ppg_projection_RW']
    player_salaries['points_LR'] = player_salaries['Prediction_LR']
    player_salaries['points_XGB'] = player_salaries['Prediction_XGB']

    # Sort by 'points_LR' in descending order
    player_salaries = player_salaries.sort_values(by='points_LR', ascending=False)

    return player_salaries
