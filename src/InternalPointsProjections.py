#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:33:47 2022

@author: nicholasrenegar
"""
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def select_features_with_lasso(player_results, expanded_columns_dict, cv=5):
    """
    Uses Lasso regression with cross-validation to select features for each position.

    Args:
    player_results (DataFrame): The combined training data for all positions.
    expanded_columns_dict (dict): Dictionary of initial features to consider by position.
    cv (int): Number of cross-validation folds.

    Returns:
    dict: Dictionary with selected features by position.
    """

    # Dictionary to store selected features and best alpha values by position
    selected_features_by_position = {}
    alpha_dict = {}

    # Loop through each position and perform Lasso feature selection
    for position, features in expanded_columns_dict.items():
        #print(f"Processing position: {position}")

        # Subset the data for the current position
        position_data = player_results[player_results['Position'] == position].copy()

        # Drop rows with missing values in the feature set
        position_data = position_data.dropna(subset=features)
        
        if position_data.empty:
            #print(f"No data available for position {position} after dropping missing values.")
            continue
        
        # Standardize the feature set
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(position_data[features])
        y = position_data['FPTS']
        
        # Apply LassoCV for automatic alpha selection
        lasso_cv = LassoCV(cv=cv, random_state=42)
        lasso_cv.fit(X_scaled, y)
        
        # Get the best alpha from LassoCV
        best_alpha = lasso_cv.alpha_
        alpha_dict[position] = best_alpha
        #print(f"Best alpha for {position}: {best_alpha}")
        
        # Extract features with non-zero coefficients
        selected_features = [feature for feature, coef in zip(features, lasso_cv.coef_) if coef != 0]

        # Save selected features to the dictionary
        selected_features_by_position[position] = selected_features
        #print(f"Selected features for {position}: {selected_features}")

    return selected_features_by_position, alpha_dict


def live_points_prediction(player_salaries, current_date, path_to_proj):
    """
    Generates internal points projections based on historical data.

    Args:
    date_string_list (list): List of date strings for which to generate projections.
    path_to_proj (str): Base path to the project directory.

    Returns:
    None: Saves model input files with projections.
    """

    # Initialize an empty DataFrame to hold all player results
    player_results = pd.DataFrame()

    # Get all historical dates
    full_date_string_list = pd.read_csv(f"{path_to_proj}/Model/_Mappings/DateList.csv")['Date'].tolist()

    # Loop through each date and aggregate player results
    for date_string in full_date_string_list:
        print(date_string)
        player_results_temp = pd.read_csv(f"{path_to_proj}/Model/Output/Points Prediction/ModelInput_Results{date_string}.csv")
        player_results_temp['DateString'] = date_string
        player_results = pd.concat([player_results, player_results_temp], ignore_index=True)

    # Handle missing values and feature engineering for historical data
    player_results['std_dev_RW'] = player_results.get('std_dev_RW', 0)
    player_results['StdDev'] = player_results['std_dev_RW'].fillna(0)
    player_results['implied_opp_score'] = player_results['implied_team_score'] + player_results['spread']

    # Handle missing values and feature engineering for live data
    player_salaries['std_dev_RW'] = player_salaries.get('std_dev_RW', 0)
    player_salaries['StdDev'] = player_salaries['std_dev_RW'].fillna(0)
    player_salaries['implied_opp_score'] = player_salaries['implied_team_score'] + player_salaries['spread']

    # Drop marginal players
    player_results = player_results.dropna(subset=['FPTS'])

    # Define potential columns for each position
    all_columns = [
        "Salary", "AvgPointsPerGame", "ppg_projection_DFF", "L5_ppg_avg", 
        "spread", "implied_team_score", "L5_dvp_rank", "ppg_projection_FSP", "ppg_projection_RW", 
        "Is_Dome", "Temp", "WindSpeed", "Pressure", "Humidity", "Is_Snow", "Is_Rain", 
        "WindSpeedStadium", "RainStadium", "SnowStadium", "ExtremeWeather", "std_dev_RW", 
        "implied_opp_score"
    ]

    expanded_columns_dict = {
        "QB": all_columns,
        "DST": all_columns,
        "TE": all_columns,
        "RB": all_columns,
        "WR": all_columns
    }

    expert_columns_dict = {
        "QB": expanded_columns_dict["QB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "DST": expanded_columns_dict["DST"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "TE": expanded_columns_dict["TE"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "RB": expanded_columns_dict["RB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "WR": expanded_columns_dict["WR"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"]
    }

    # Select features using Lasso
    selected_features_dict, alpha_dict = select_features_with_lasso(player_results, expert_columns_dict)

    # Loop through each position and generate projections
    for position, norm_columns in selected_features_dict.items():
        print(f"Processing position: {position}")
        position_data = player_results[player_results['Position'] == position]
        position_data = position_data.dropna(subset=norm_columns)

        # Apply PCA and scaling to the full dataset before splitting into train and test
        scaler = StandardScaler()
        pca = PCA(n_components=0.99)
        
        # Scale and apply PCA to the norm_columns
        pca_data = scaler.fit_transform(position_data[norm_columns])
        pca_scores = pca.fit_transform(pca_data)
        
        # Manually rename the PCA columns
        pca_columns = [f'PC{i+1}' for i in range(pca_scores.shape[1])]
        position_data[pca_columns] = pca_scores
        
        # Now you can use these PCA columns in your models
        ind_var_columns = pca_columns

        # Create out-of-sample predictions for each game day
        print(f"Predicting Points for date: {current_date}")
        train_data = position_data[position_data['DateString'].astype(str) != str(current_date)]
        test_data = player_salaries[player_salaries['Position'] == position].copy()
        
        # Fill missing values in test_data with the mean values from train_data
        test_data[norm_columns] = test_data[norm_columns].fillna(train_data[norm_columns].mean())

        # Apply the same PCA transformation to the test_data
        test_data_pca = scaler.transform(test_data[norm_columns])
        test_data_pca = pca.transform(test_data_pca)
        test_data[pca_columns] = test_data_pca

        # Linear Regression
        lr = LinearRegression()
        lr.fit(train_data[ind_var_columns], train_data['FPTS'])
        test_data['Prediction_LR'] = lr.predict(test_data[ind_var_columns])
        
        # Ridge Regression
        ridge = Ridge(alpha=alpha_dict[position])
        ridge.fit(train_data[ind_var_columns], train_data['FPTS'])
        test_data['Prediction_Ridge'] = ridge.predict(test_data[ind_var_columns])
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha_dict[position])
        lasso.fit(train_data[ind_var_columns], train_data['FPTS'])
        test_data['Prediction_Lasso'] = lasso.predict(test_data[ind_var_columns])
        
        # Random Forest
        rf = RandomForestRegressor()
        rf.fit(train_data[ind_var_columns], train_data['FPTS'])
        test_data['Prediction_RF'] = rf.predict(test_data[ind_var_columns])
        
        # XGBoost
        dtrain = xgb.DMatrix(train_data[ind_var_columns], label=train_data['FPTS'])
        dtest = xgb.DMatrix(test_data[ind_var_columns])
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5
        }
        bst = xgb.train(params, dtrain, num_boost_round=100)
        test_data['Prediction_XGB'] = bst.predict(dtest)
        
        # Add test predictions to the player_salaries DataFrame
        for col in ['Prediction_LR', 'Prediction_Ridge', 'Prediction_Lasso', 'Prediction_RF', 'Prediction_XGB']:
            player_salaries.loc[player_salaries['Position'] == position, col] = test_data[col]

    return player_salaries


def quick_backtest_XGB(date_string_list, path_to_proj, num_features=8):
    """
    Generates internal points projections based on historical data.

    Args:
    date_string_list (list): List of date strings for which to generate projections.
    path_to_proj (str): Base path to the project directory.

    Returns:
    None: Saves model input files with projections.
    """

    # Initialize an empty DataFrame to hold all player results
    player_results = pd.DataFrame()

    # Loop through each date and aggregate player results
    for date_string in date_string_list:
        print(date_string)
        player_results_temp = pd.read_csv(f"{path_to_proj}/Model/Output/Points Prediction/ModelInput_Results{date_string}.csv")
        player_results_temp['DateString'] = date_string
        player_results = pd.concat([player_results, player_results_temp], ignore_index=True)

    # Drop marginal players
    player_results = player_results.dropna(subset=['FPTS'])

    # Define potential columns for each position
    all_columns = [
        "Salary", "AvgPointsPerGame", "ppg_projection_DFF", "L5_ppg_avg", 
        "spread", "implied_team_score", "L5_dvp_rank", "ppg_projection_FSP", "ppg_projection_RW", 
        "Is_Dome", "Temp", "WindSpeed", "Pressure", "Humidity", "Is_Snow", "Is_Rain", 
        "WindSpeedStadium", "RainStadium", "SnowStadium", "ExtremeWeather", "std_dev_RW", 
        "implied_opp_score"
    ]

    expanded_columns_dict = {
        "QB": all_columns,
        "DST": all_columns,
        "TE": all_columns,
        "RB": all_columns,
        "WR": all_columns
    }

    expert_columns_dict = {
        "QB": expanded_columns_dict["QB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "DST": expanded_columns_dict["DST"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "TE": expanded_columns_dict["TE"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "RB": expanded_columns_dict["RB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "WR": expanded_columns_dict["WR"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"]
    }

    # Select features using Lasso
    selected_features_dict, alpha_dict = select_features_with_lasso(player_results, expert_columns_dict)

    # Perform feature selection across all dates for training data
    for position, feature_columns in expanded_columns_dict.items():
        print(f"Selecting features for position: {position}")

        # Combine training data across all dates for feature selection
        combined_train_data = player_results[player_results['Position'] == position].copy()
        combined_train_data = combined_train_data.dropna(subset=feature_columns)

        # Prepare DMatrix for XGBoost
        dtrain = xgb.DMatrix(combined_train_data[feature_columns], label=combined_train_data['FPTS'])

        # Train initial XGBoost model to get feature importance
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5
        }
        bst = xgb.train(params, dtrain, num_boost_round=100)

        # Get feature importance and select top features
        importance_dict = bst.get_score(importance_type='weight')
        sorted_features = sorted(importance_dict, key=importance_dict.get, reverse=True)

        # Select top features based on num_features parameter
        if num_features:
            selected_features = sorted_features[:num_features]
        else:
            selected_features = sorted_features
        
        selected_features_dict[position] = selected_features
        print(f"Selected features for {position}: {selected_features}")

    # Prepare to collect results
    points_predictions = pd.DataFrame()
    
    # Loop through each position and generate projections using selected features
    for position, selected_features in selected_features_dict.items():
        #print(f"Processing position: {position} with selected features")

        position_data = player_results[player_results['Position'] == position]
    
        # Time-based split for training and testing
        for date_string in date_string_list:
            #print(f"Predicting for date: {date_string}")
            train_data = position_data[position_data['DateString'] != date_string].copy()
            test_data = position_data[position_data['DateString'] == date_string].copy()

            # Ensure consistent handling of missing values
            train_data = train_data.dropna(subset=selected_features)
            test_data[selected_features] = test_data[selected_features].fillna(train_data[selected_features].mean())

            # Apply scaling and PCA
            scaler = StandardScaler()
            train_data_scaled = scaler.fit_transform(train_data[selected_features])
            test_data_scaled = scaler.transform(test_data[selected_features])

            pca = PCA(n_components=0.99)
            train_data_pca = pca.fit_transform(train_data_scaled)
            test_data_pca = pca.transform(test_data_scaled)

            dtrain_selected = xgb.DMatrix(train_data_pca, label=train_data['FPTS'])
            dtest_selected = xgb.DMatrix(test_data_pca)

            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.005,
                'max_depth': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5
            }
            bst = xgb.train(params, dtrain_selected, num_boost_round=1000)

            # Prediction
            test_data['Prediction_XGB_New'] = bst.predict(dtest_selected)
    
            points_predictions = pd.concat([points_predictions, test_data], ignore_index=True)

    # Calculate and print correlations by position
    print("\nCorrelations by position:")
    for position in selected_features_dict.keys():
        position_data = points_predictions[points_predictions['Position'] == position]
        if not position_data.empty:
            print(f"\nPosition: {position}")
            print(f"ppg_projection_DFF: {position_data['FPTS'].corr(position_data['ppg_projection_DFF'])}")
            print(f"ppg_projection_FSP: {position_data['FPTS'].corr(position_data['ppg_projection_FSP'])}")
            print(f"ppg_projection_RW: {position_data['FPTS'].corr(position_data['ppg_projection_RW'])}")
            print(f"Prediction_XGB new: {position_data['FPTS'].corr(position_data['Prediction_XGB_New'])}")
    
    points_predictions.to_csv(os.path.join(path_to_proj, "Model/Output/Backtest_Results.csv"), index=False)



def backtest_points_projections(date_string_list, path_to_proj):
    """
    Generates internal points projections based on historical data.

    Args:
    date_string_list (list): List of date strings for which to generate projections.
    path_to_proj (str): Base path to the project directory.

    Returns:
    None: Saves model input files with projections.
    """

    # Initialize an empty DataFrame to hold all player results
    player_results = pd.DataFrame()

    # Loop through each date and aggregate player results
    for date_string in date_string_list:
        print(date_string)
        player_results_temp = pd.read_csv(f"{path_to_proj}/Model/Output/Points Prediction/ModelInput_Results{date_string}.csv")
        player_results_temp['DateString'] = date_string
        player_results = pd.concat([player_results, player_results_temp], ignore_index=True)

    # Handle missing values and feature engineering
    if 'std_dev_RW' not in player_results.columns:
        player_results['std_dev_RW'] = 0
    player_results['std_dev_RW'] = player_results['std_dev_RW'].fillna(0)
    player_results['implied_opp_score'] = player_results['implied_team_score'] + player_results['spread']

    # Drop marginal players
    player_results = player_results.dropna(subset=['FPTS'])

    # Define potential columns for each position
    all_columns = [
        "Salary", "AvgPointsPerGame", "ppg_projection_DFF", "L5_ppg_avg", 
        "spread", "implied_team_score", "L5_dvp_rank", "ppg_projection_FSP", "ppg_projection_RW", 
        "Is_Dome", "Temp", "WindSpeed", "Pressure", "Humidity", "Is_Snow", "Is_Rain", 
        "WindSpeedStadium", "RainStadium", "SnowStadium", "ExtremeWeather", "std_dev_RW", 
        "implied_opp_score"
    ]

    expanded_columns_dict = {
        "QB": all_columns,
        "DST": all_columns,
        "TE": all_columns,
        "RB": all_columns,
        "WR": all_columns
    }

    expert_columns_dict = {
        "QB": expanded_columns_dict["QB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "DST": expanded_columns_dict["DST"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "TE": expanded_columns_dict["TE"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "RB": expanded_columns_dict["RB"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"],
        "WR": expanded_columns_dict["WR"] + ["ExpertAverageScore", "ExpertNumMentions", "WaiverAverageScore", "WaiverNumMentions"]
    }

    # Select features using Lasso
    selected_features_dict, alpha_dict = select_features_with_lasso(player_results, expert_columns_dict)

    points_predictions = pd.DataFrame()

    # Loop through each position and generate projections
    for position, norm_columns in selected_features_dict.items():
        print(f"Processing position: {position}")
        position_data = player_results[player_results['Position'] == position]
        position_data = position_data.dropna(subset=norm_columns)

        # Apply PCA and scaling to the full dataset before splitting into train and test
        scaler = StandardScaler()
        pca = PCA(n_components=0.99)
        
        # Scale and apply PCA to the norm_columns
        pca_data = scaler.fit_transform(position_data[norm_columns])
        pca_scores = pca.fit_transform(pca_data)
        
        # Manually rename the PCA columns
        pca_columns = [f'PC{i+1}' for i in range(pca_scores.shape[1])]
        position_data[pca_columns] = pca_scores
        
        # Now you can use these PCA columns in your models
        ind_var_columns = pca_columns

        # Create out-of-sample predictions for each game day
        for date_string in date_string_list:
            print(f"Predicting for date: {date_string}")
            train_data = position_data[position_data['DateString'] != date_string]
            test_data = position_data[position_data['DateString'] == date_string].copy()  # Use .copy() to avoid SettingWithCopyWarning
            
            # Linear Regression
            lr = LinearRegression()
            lr.fit(train_data[ind_var_columns], train_data['FPTS'])
            test_data.loc[:, 'Prediction_LR'] = lr.predict(test_data[ind_var_columns])
            
            # Ridge Regression
            ridge = Ridge(alpha=alpha_dict[position])
            ridge.fit(train_data[ind_var_columns], train_data['FPTS'])
            test_data.loc[:, 'Prediction_Ridge'] = ridge.predict(test_data[ind_var_columns])
            
            # Lasso Regression
            lasso = Lasso(alpha=alpha_dict[position])
            lasso.fit(train_data[ind_var_columns], train_data['FPTS'])
            test_data.loc[:, 'Prediction_Lasso'] = lasso.predict(test_data[ind_var_columns])
            
            # Random Forest
            rf = RandomForestRegressor()
            rf.fit(train_data[ind_var_columns], train_data['FPTS'])
            test_data.loc[:, 'Prediction_RF'] = rf.predict(test_data[ind_var_columns])
            
            # XGBoost
            dtrain = xgb.DMatrix(train_data[ind_var_columns], label=train_data['FPTS'])
            dtest = xgb.DMatrix(test_data[ind_var_columns])
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.1,
                'max_depth': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5
            }
            bst = xgb.train(params, dtrain, num_boost_round=100)
            test_data.loc[:, 'Prediction_XGB'] = bst.predict(dtest)
            
            # Add test predictions to the final DataFrame
            points_predictions = pd.concat([points_predictions, test_data], ignore_index=True)

        # Display correlation results
        print("Correlations with actual points:")
        print(f"ppg_projection_DFF: {points_predictions['FPTS'].corr(points_predictions['ppg_projection_DFF'])}")
        print(f"ppg_projection_FSP: {points_predictions['FPTS'].corr(points_predictions['ppg_projection_FSP'])}")
        print(f"ppg_projection_RW: {points_predictions['FPTS'].corr(points_predictions['ppg_projection_RW'])}")
        print(f"Prediction_LR: {points_predictions['FPTS'].corr(points_predictions['Prediction_LR'])}")
        print(f"Prediction_XGB: {points_predictions['FPTS'].corr(points_predictions['Prediction_XGB'])}")
        print(f"Prediction_Ridge: {points_predictions['FPTS'].corr(points_predictions['Prediction_Ridge'])}")
        print(f"Prediction_Lasso: {points_predictions['FPTS'].corr(points_predictions['Prediction_Lasso'])}")
        print(f"Prediction_RF: {points_predictions['FPTS'].corr(points_predictions['Prediction_RF'])}")



    



