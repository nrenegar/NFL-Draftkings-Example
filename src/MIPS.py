#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:02:20 2022

@author: nicholasrenegar
"""
import subprocess
import os

def run_julia_mips(date_string_list, points_system_list, num_lineups_list, num_overlap_list, max_appearances_list, alpha_list, path_to_proj):
    # Define the path to the Julia executable
    julia_executable = "/Applications/Julia-0.5.app/Contents/Resources/julia/bin/julia"  # Update this path as needed
    
    # Define the path to the Julia script
    julia_script_path = "/Users/nicholasrenegar/Desktop/Github/NFL-Draftkings/src/MIPS.jl"
    
    # Ensure the Julia script exists
    if not os.path.exists(julia_script_path):
        print(f"Error: The Julia script {julia_script_path} does not exist.")
        return
    
    # Convert the date strings to integers
    date_list = [int(date) for date in date_string_list]

    # Convert the lists to Julia array format as strings
    date_list_str = "[" + ','.join(map(str, date_list)) + "]"
    points_system_list_str = "[" + ','.join(f'"{x}"' for x in points_system_list) + "]"
    num_lineups_list_str = "[" + ','.join(map(str, num_lineups_list)) + "]"
    num_overlap_list_str = "[" + ','.join(map(str, num_overlap_list)) + "]"
    max_appearances_list_str = "[" + ','.join(map(str, max_appearances_list)) + "]"
    alpha_list_str = "[" + ','.join(map(str, alpha_list)) + "]"
    
    print(date_list_str,
        points_system_list_str,
        num_lineups_list_str,
        num_overlap_list_str,
        max_appearances_list_str,
        alpha_list_str,
        path_to_proj)
    # Build the command to run the Julia script
    command = [
        julia_executable, 
        julia_script_path,
        date_list_str,
        points_system_list_str,
        num_lineups_list_str,
        num_overlap_list_str,
        max_appearances_list_str,
        alpha_list_str,
        path_to_proj
    ]
    
    # Run the Julia script using subprocess.Popen to print output in real-time
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end='')  # Print each line as it is generated
            
            for line in process.stderr:
                print(line, end='')  # Print each error line as it is generated
            
            process.wait()  # Wait for the process to complete
    except Exception as e:
        print(f"An error occurred while running the Julia script: {e}")
