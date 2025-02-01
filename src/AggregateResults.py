#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 23:43:01 2022

@author: nicholasrenegar
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess

def aggregate_and_plot_results(date_string, path_to_proj, num_lineups=150):
    """
    Aggregates contest results and displays charts.

    Args:
    date_string (str): The date string used for the file names.
    num_lineups (int): The number of top lineups to consider.
    path_to_proj (str): The base path to the project directory.

    Returns:
    None: Displays charts and writes aggregated results to CSV.
    """

    # Set working directory
    os.chdir(path_to_proj)
    
    #=====================================================================#
    #            Import All Aggregate Results and Combine                 #
    #=====================================================================#
    lineup_filenames = [f for f in os.listdir(f"{path_to_proj}/Model/Output/Results/") 
                        if "Lineup" in f and "_Top" in f]
    
    lineups = pd.DataFrame()

    for filename in lineup_filenames:
        current_lineup = pd.read_csv(f"{path_to_proj}/Model/Output/Results/{filename}")
        current_lineup['Filename'] = filename
        lineups = pd.concat([lineups, current_lineup], ignore_index=True)

    # Drop rows where ROI is missing
    lineups = lineups.dropna(subset=['ROI'])

    #=====================================================================#
    #                 Get Aggregate Results by Parameters                 #
    #=====================================================================#
    # Limit to top n lineups
    lineups = lineups[lineups['iteration'] <= num_lineups]

    # Aggregate ROI by different parameters
    agg_results = lineups.groupby(['points_system', 'num_overlap', 'max_appearances', 'alpha', 'num_lineups'])['ROI'].mean().reset_index()
    agg_results_points_system = lineups.groupby('points_system')['ROI'].mean().reset_index()
    agg_results_num_overlap = lineups.groupby('num_overlap')['ROI'].mean().reset_index()
    agg_results_max_appearances = lineups.groupby('max_appearances')['ROI'].mean().reset_index()
    agg_results_num_lineups = lineups.groupby('num_lineups')['ROI'].mean().reset_index()
    agg_results_alpha = lineups.groupby('alpha')['ROI'].mean().reset_index()

    lineups['Params'] = lineups[['points_system', 'num_overlap', 'max_appearances', 'num_lineups', 'alpha']].astype(str).agg(' '.join, axis=1)
    agg_results_date = lineups.groupby(['Params', 'Date'])['ROI'].mean().unstack().reset_index()

    #=====================================================================#
    #                           Output Results                            #
    #=====================================================================#
    agg_results_date.to_csv(f"{path_to_proj}/Model/CumulativeResults.csv", index=False)
    lineups.to_csv(f"{path_to_proj}/Model/Output/Results/CumulativeLineups.csv", index=False)

    #=====================================================================#
    #                     Plot ROI by Lineup Number                       #
    #=====================================================================#
    lineups['ROI_Adj'] = np.minimum(lineups['ROI'], 1000)

    plt.figure(figsize=(10, 6))
    plt.scatter(lineups['iteration'], lineups['ROI'], color='blue', label='ROI')
    
    # Linear Regression Line
    slope, intercept, r_value, p_value, std_err = linregress(lineups['iteration'], lineups['ROI'])
    plt.plot(lineups['iteration'], intercept + slope * lineups['iteration'], 'r', label=f'Linear fit (slope: {slope:.2f})')
    
    # Lowess Line
    lowess_line = lowess(lineups['ROI'], lineups['iteration'], frac=0.1)
    plt.plot(lowess_line[:, 0], lowess_line[:, 1], 'g', label='Lowess fit')

    # Draw a dotted line at y = 1
    plt.axhline(y=1, color='black', linestyle='--', label='ROI = 1')

    plt.xlabel('Iteration')
    plt.ylabel('ROI')
    plt.title('ROI vs Iteration')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()

    #=====================================================================#
    #        Analyze ROI by Lineup Number and Kelly Criterion             #
    #=====================================================================#

    #Get ROI and log returns based on optimal kelly criteria
    num_lineups_roi = analyze_num_lineups(lineups)
    
    # Find the best ROI
    best_roi = agg_results['ROI'].max()
    best_parameters = agg_results[agg_results['ROI'] == best_roi]
    
    # Filter lineups to only those matching the best ROI parameters
    lineups_best_roi = pd.DataFrame()
    for _, row in best_parameters.iterrows():
        matching_lineups = lineups[
            (lineups['points_system'] == row['points_system']) &
            (lineups['num_overlap'] == row['num_overlap']) &
            (lineups['max_appearances'] == row['max_appearances']) &
            (lineups['alpha'] == row['alpha']) &
            (lineups['num_lineups'] == row['num_lineups'])
        ]
        lineups_best_roi = pd.concat([lineups_best_roi, matching_lineups], ignore_index=True)
    
    # Assuming 'analyze_num_lineups' calculates ROI and log returns based on optimal kelly criteria
    num_lineups_roi_best = analyze_num_lineups(lineups_best_roi)

    return agg_results_date


def optimal_kelly_criterion(rois):
    """
    Calculates the optimal bet size using the Kelly criterion to maximize the average log utility across all given ROIs.
    """
    def avg_negative_log_utility(bet_size):
        utility = np.log(1 - bet_size + bet_size * rois)
        return -np.nanmean(utility)  # Use nanmean to ignore NaN results from invalid operations

    # Optimize to find the bet size that minimizes the negative average log utility
    res = minimize_scalar(avg_negative_log_utility, bounds=(0, 1), method='bounded')
    if res.success:
        return max(0, min(res.x, 1)), -res.fun  # Ensure bet size is within [0, 1] and return positive average log utility
    else:
        return 0, float('nan')

def analyze_num_lineups(lineups):
    results = []
    for num_lineups in range(10, 151):
        # Calculate average ROI by date for the current num_lineups
        avg_roi_by_date = lineups[lineups['iteration'] <= num_lineups].groupby('Date')['ROI'].mean()

        # Calculate the optimal bet size and corresponding average log utility
        optimal_bet_size, average_log_utility = optimal_kelly_criterion(avg_roi_by_date.values)

        # Calculate average ROI across all dates
        avg_roi_by_date_bet_size = 1 - optimal_bet_size + avg_roi_by_date*optimal_bet_size
        avg_roi = avg_roi_by_date_bet_size.mean()

        results.append({
            'NumLineups': num_lineups,
            'Average ROI': avg_roi,
            'Optimal Bet Size': optimal_bet_size,
            'Average Log Utility': average_log_utility
        })
    
    return pd.DataFrame(results)
