#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:57:20 2022

@author: nicholasrenegar
"""

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import string
import os


def preprocess_name(name):
    # Remove punctuation and convert to lowercase
    return ''.join(char for char in name if char not in string.punctuation).lower()

def test_match_names(mapping_name):
    # Split into exact matches and potential mismatches
    exact_matches = mapping_name[mapping_name['Name'] == mapping_name['Final']]
    possible_mismatches = mapping_name[mapping_name['Name'] != mapping_name['Final']]
    
    # Create a list of original and preprocessed names for exact matches
    exact_names_original = exact_matches['Final'].tolist()
    exact_names_preprocessed = exact_matches['Final'].apply(preprocess_name).tolist()
    
    # Function to find the closest match using Levenshtein distance after preprocessing
    def find_closest_match(name):
        name_preprocessed = preprocess_name(name)
        closest_match_preprocessed, score = process.extractOne(
            name_preprocessed, 
            exact_names_preprocessed, 
            scorer=fuzz.ratio
        )
        
        # Find the index of the preprocessed match to get the original formatted name
        closest_match_index = exact_names_preprocessed.index(closest_match_preprocessed)
        closest_match_original = exact_names_original[closest_match_index]
        
        return closest_match_original, score
    
    # Apply the function to get suggested names and distances
    possible_mismatches[['Suggested_Final', 'Distance_Score']] = possible_mismatches['Name'].apply(
        lambda x: pd.Series(find_closest_match(x))
    )

    return possible_mismatches


def find_closest_match(name, reference_names_preprocessed, reference_names_original):
    # Preprocess the input name
    name_preprocessed = preprocess_name(name)
    
    # Find the closest match in the preprocessed reference names
    closest_match_preprocessed, score = process.extractOne(
        name_preprocessed, 
        reference_names_preprocessed, 
        scorer=fuzz.ratio
    )
    
    # Find the index of the preprocessed match to get the original formatted name
    closest_match_index = reference_names_preprocessed.index(closest_match_preprocessed)
    closest_match_original = reference_names_original[closest_match_index]
    
    return closest_match_original, score

def remap_player_names(column, path_to_proj):
    mapping_name_path = os.path.join(path_to_proj, "Model/_Mappings/mappingName.csv")
    mapping_name = pd.read_csv(mapping_name_path)
    mapping_name['Name'] = mapping_name['Name'].astype(str)
    mapping_name['Final'] = mapping_name['Final'].astype(str)

    # Create a dictionary from mapping_names to quickly find exact matches
    mapping_dict = pd.Series(mapping_name['Final'].values, index=mapping_name['Name']).to_dict()
 
    # Prepare exact match and preprocessed names for closest match lookup
    exact_names_original = mapping_name['Final'].tolist()
    exact_names_preprocessed = mapping_name['Final'].apply(preprocess_name).tolist()
    
    remapped_names = {}
    new_mappings = []


    # Process each name in the input column
    for name in column:
        if name in mapping_dict:
            # If name exists in the mapping dictionary, use the mapped name
            new_name = mapping_dict[name]
        else:
            # Find the closest match if no exact mapping exists
            new_name, score = find_closest_match(name, exact_names_preprocessed, exact_names_original)
            
            # Only print and add to new mappings if the name is not already in mapping_dict
            if name not in mapping_dict.values():
                print(f"Suggested remapping for '{name}': '{new_name}' (Score: {score})")
                new_mappings.append((name, new_name))

        remapped_names[name] = new_name

    # Apply the remapped names to the input column
    remapped_column = column.map(remapped_names)

    # Add new mappings to the mapping_names DataFrame
    if new_mappings:
        new_mappings_df = pd.DataFrame(new_mappings, columns=['Name', 'Final'])
        mapping_name = pd.concat([mapping_name, new_mappings_df], ignore_index=True)
    
    # Save updated mapping_names to CSV
    mapping_name.to_csv(f"{path_to_proj}/Model/_Mappings/mappingName.csv", index=False)

    return remapped_column


