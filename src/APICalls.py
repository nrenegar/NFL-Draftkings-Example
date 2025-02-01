#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:56:44 2022

@author: nicholasrenegar
"""


import requests
import pandas as pd
from datetime import datetime
from pytz import timezone


weather_api_key = "*****************************"
  
  
##################################################
#################Weather Info###################
##################################################

def fetch_weather(api_key, lat, lon, datetime_str):
    base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S").astimezone(timezone('UTC'))
    dt = int(datetime_obj.timestamp())
    
    params = {
        'lat': lat,
        'lon': lon,
        'dt': dt,
        'appid': api_key,
        'units': 'imperial'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def add_weather_info(player_salaries, api_key=weather_api_key):
    # Get unique games for weather API Calls
    game_list = player_salaries[['Game_DateTime', 'Latitude', 'Longitude']].drop_duplicates().dropna()
    
    # Initialize weather-related columns
    game_list['Temp'] = None
    game_list['WindSpeed'] = None
    game_list['Pressure'] = None
    game_list['Humidity'] = None
    game_list['Is_Snow'] = None
    game_list['Is_Rain'] = None
    
    # Fetch weather data for each unique game
    for index, row in game_list.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        datetime_str = row['Game_DateTime']
        weather_data = fetch_weather(api_key, lat, lon, datetime_str)
        
        weather = weather_data['data'][0]
        game_list.at[index, 'Temp'] = weather.get('temp')
        game_list.at[index, 'WindSpeed'] = weather.get('wind_speed')
        game_list.at[index, 'Pressure'] = weather.get('pressure')
        game_list.at[index, 'Humidity'] = weather.get('humidity')
        # Handle nested dicts for snow and rain
        snow_data = weather.get('snow', {})
        rain_data = weather.get('rain', {})
        
        game_list.at[index, 'Is_Snow'] = 1 if snow_data.get('1h', 0) > 0 else 0
        game_list.at[index, 'Is_Rain'] = 1 if rain_data.get('1h', 0) > 0 else 0
    
    # Merge the weather information back into the player_salaries DataFrame
    player_salaries = pd.merge(player_salaries, game_list, on=['Game_DateTime', 'Latitude', 'Longitude'], how='left')
    
    # Adjust stadium weather conditions
    player_salaries['WindSpeedStadium'] = player_salaries['WindSpeed'] * (1 - player_salaries['Is_Dome'])
    player_salaries['RainStadium'] = player_salaries['Is_Rain'] * (1 - player_salaries['Is_Dome'])
    player_salaries['SnowStadium'] = player_salaries['Is_Snow'] * (1 - player_salaries['Is_Dome'])
    
    # Define extreme weather conditions
    player_salaries['ExtremeWeather'] = player_salaries.apply(
        lambda row: 1 if row['Is_Dome'] == 0 and (row['WindSpeed'] > 15 or row['Temp'] < 35) else 0,
        axis=1
    )
    
    return player_salaries


##################################################
#################API Calls Main###################
##################################################
def api_calls(player_salaries):
    #Add Weather Data
    player_salaries = add_weather_info(player_salaries)
    
    return player_salaries