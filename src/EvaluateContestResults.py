#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:12:33 2022

@author: nicholasrenegar
"""

import pandas as pd
import os
import re

def parse_first_rank(rank):
    # Split the rank string on "-" and process both parts
    parts = rank.split("-")
    return int(re.sub(r'\D', '', parts[0]))  # Extract the numeric part

def parse_last_rank(rank):
    # Split the rank string on "-" and process both parts
    parts = rank.split("-")
    if len(parts) > 1:
        return int(re.sub(r'\D', '', parts[1]))  # Extract the numeric part
    else:
        return int(re.sub(r'\D', '', parts[0]))  # Extract the numeric part

def check_prizes_and_revenue(competitor_results, prizes, entry_fee):
    """
    Checks if the total payout is within the expected range of 85% - 95% of the total entry fees.

    Args:
    competitor_results (DataFrame): DataFrame containing the results of the contest.
    prizes (DataFrame): DataFrame containing the prize structure of the contest.
    entry_fee (float): The entry fee for the contest.

    Returns:
    None: Prints a warning message if the payout percentage is outside the expected range.
    """

    # Calculate the total DraftKings revenue
    total_entries = competitor_results.shape[0]
    total_revenue = entry_fee * total_entries

    # Calculate the number of winners for each prize based on LastRank
    prizes['NumberOfWinners'] = prizes['LastRank'].diff().fillna(prizes['LastRank']).astype(int)

    # Calculate the total payout
    total_payout = (prizes['Prize_Amount'] * prizes['NumberOfWinners']).sum()

    # Calculate the payout percentage
    payout_percentage = (total_payout / total_revenue) * 100

    # Check if the payout percentage is outside the 80% - 90% range
    if payout_percentage < 80 or payout_percentage > 90:
        print(f"Warning: The payout percentage is {payout_percentage:.2f}%, which is outside the expected range. Please check the prizes and entry fee calculations.")
    
    # Print the total revenue and payout for verification
    #print(f"Total Revenue: ${total_revenue:.2f}")
    #print(f"Total Payout: ${total_payout:.2f}")



def evaluate_draftkings_results(date_string, path_to_proj, num_lineups=150):
    # Set working directory
    os.chdir(path_to_proj)

    # Import name mappings
    mapping_name = pd.read_csv(f"{path_to_proj}/Model/_Mappings/mappingName.csv")
    
    # Get Contest Results
    competitor_results = pd.read_csv(f"{path_to_proj}/Data/Draftkings Results/contest-standings-{date_string}_Top.csv").iloc[:, :6]
    player_results = pd.read_csv(f"{path_to_proj}/Data/Draftkings Results/contest-standings-{date_string}_Top.csv").iloc[:, 7:11]
    player_results = player_results[player_results['Player'] != '']
    
    # Get Prizes
    prizes = pd.read_csv(f"{path_to_proj}/Data/Draftkings Results/Prizes{date_string}_Top.csv")
    entry_fee_str = prizes['Entry'][0]
    entry_fee = float(entry_fee_str.replace('$', '').replace(',', ''))
    prizes['Prize_Amount'] = prizes['Prizes'].str.replace('[$,]', '', regex=True).astype(float)
    prizes['ROI'] = (prizes['Prize_Amount']) / entry_fee

    # Calculate Rank Range for each prize tier
    prizes['RankSplit'] = prizes['Rank'].str.split("-")
    prizes['FirstRank'] = prizes['Rank'].apply(parse_first_rank)
    prizes['LastRank'] = prizes['Rank'].apply(parse_last_rank)

    # Check prizes and revenue
    check_prizes_and_revenue(competitor_results, prizes, entry_fee)

    # Import all lineups
    lineup_filenames = [f for f in os.listdir(f"{path_to_proj}/Model/Output/Lineups_Internal/") if date_string in f]
    lineups = pd.DataFrame(columns=['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST', 
                                    'points_system', 'num_lineups', 'num_overlap', 'max_appearances', 
                                    'alpha', 'iteration', 'Objective', 'Parameters', 'FPTS'])
    
    for filename in lineup_filenames:
        current_lineup = pd.read_csv(f"{path_to_proj}/Model/Output/Lineups_Internal/{filename}")
        current_lineup['DST'] = current_lineup['DST'].astype(str)
        current_lineup['Parameters'] = filename
        current_lineup['FPTS'] = 0
        
        for i in range(current_lineup.shape[0]):
            fpts = 0
            for j in range(9):
                player_name = current_lineup.iloc[i, j]
                # Use .loc for boolean indexing
                remapped_name = mapping_name.loc[mapping_name['Name'] == player_name, 'Final'].values
                
                if len(remapped_name) > 0:
                    current_lineup.iloc[i, j] = remapped_name[0]
                
                    if player_results[player_results['Player'] == remapped_name[0]].shape[0] >= 1:
                        fpts += player_results[player_results['Player'] == remapped_name[0]].iloc[0, 3]
            
            current_lineup.at[i, 'FPTS'] = fpts
        
        lineups = pd.concat([lineups, current_lineup], ignore_index=True)
    
    lineups.dropna(inplace=True)
    
    # Calculate Retrospective Winnings for Each Lineup
    lineups['Rank'] = lineups['FPTS'].apply(lambda x: (competitor_results['Points'] < x).idxmax() + 1)
    lineups['ROI'] = 0
    lineups['Date'] = date_string

    for i in range(lineups.shape[0]):
        fpts = lineups.at[i, 'FPTS']
        
        # Find all competitors with the same FPTS to account for ties
        tied_competitors = competitor_results[competitor_results['Points'] == fpts].copy()  # Use .copy() to avoid SettingWithCopyWarning
        if tied_competitors.empty:
            for j in range(prizes.shape[0]):
                if lineups.at[i, 'ROI'] == 0 and lineups.at[i, 'Rank'] <= prizes.at[j, 'LastRank']:
                    lineups.at[i, 'ROI'] = prizes.at[j, 'ROI']
        
        else:
            # Set the ROI for tied competitors from Prizes where Rank matches
            tied_competitors["AdjRank"] = tied_competitors.index+1
            tied_competitors['ROI'] = tied_competitors['AdjRank'].apply(
                lambda rank: prizes.loc[(prizes['FirstRank'] <= rank) & (prizes['LastRank'] >= rank), 'ROI'].values[0]
                if not prizes.loc[(prizes['FirstRank'] <= rank) & (prizes['LastRank'] >= rank)].empty else 0
            )

            # Calculate the average ROI for the tied competitors
            avg_roi = tied_competitors['ROI'].mean()
                        
            # Set the ROI in the lineups dataframe to the average ROI
            lineups.at[i, 'ROI'] = avg_roi
        
    # Aggregate Results by Parameters
    lineups_top = lineups[lineups['iteration'] <= num_lineups]
    
    agg_results = lineups_top.groupby('Parameters')['ROI'].mean().reset_index()
    #agg_results_num_lineups = lineups_top.groupby('num_lineups')['ROI'].mean().reset_index()
    #agg_results_num_overlap = lineups_top.groupby('num_overlap')['ROI'].mean().reset_index()
    #agg_results_max_appearances = lineups_top.groupby('max_appearances')['ROI'].mean().reset_index()
    #agg_results_alpha = lineups_top.groupby('alpha')['ROI'].mean().reset_index()

    # Output Results
    lineups.to_csv(f"{path_to_proj}/Model/Output/Results/LineupResults{date_string}_Top.csv", index=False)
    agg_results.to_csv(f"{path_to_proj}/Model/Output/Results/AggResults{date_string}_Top.csv", index=False)
    
    # Evaluate Points Prediction Models
    model_input = pd.read_csv(f"{path_to_proj}/Model/Output/Model Input/Model Input{date_string}.csv")
    model_input_results = model_input.merge(player_results[['Player', 'FPTS', '%Drafted']], 
                                            left_on='Name', right_on='Player', how='left')
    
    for position in model_input_results['Position'].unique():
        position_data = model_input_results[model_input_results['Position'] == position].dropna(subset=['FPTS'])
        print(f"{position} correlations:")
        print(f"DFF_FSP: {position_data['FPTS'].corr(position_data['points_dff_fsp'])}")
        print(f"RW: {position_data['FPTS'].corr(position_data['points_rw'])}")
        print(f"LR: {position_data['FPTS'].corr(position_data['points_LR'])}")
        print(f"XGB: {position_data['FPTS'].corr(position_data['points_XGB'])}")

    model_input_results.to_csv(f"{path_to_proj}/Model/Output/Points Prediction/ModelInput_Results{date_string}.csv", index=False)

    return agg_results



def update_date_list(date_string, path_to_proj):
    """
    Checks if a date_string exists in the date list. If not, adds it and saves to CSV.

    Args:
    date_string (str): The new date string to check and add.
    path_to_proj (str): Base path to the project directory.

    Returns:
    None: Updates the CSV file with the new date list if necessary.
    """

    # Load existing date list
    date_list_path = f"{path_to_proj}/Model/_Mappings/DateList.csv"
    full_date_string_list = pd.read_csv(date_list_path)['Date'].astype(str).tolist()

    # Check if the new date_string is already in the list
    if date_string not in full_date_string_list:
        print(f"Adding new date: {date_string} to date list.")
        full_date_string_list.append(date_string)

        # Save updated date list to CSV
        updated_date_df = pd.DataFrame({'Date': full_date_string_list})
        updated_date_df.to_csv(date_list_path, index=False)
        print(f"Updated date list saved to {date_list_path}.")