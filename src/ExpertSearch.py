#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday July 19, 2024

@author: Nicholas Renegar
"""

import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from openai import OpenAI
from src import NameMatching
from datetime import datetime, timedelta
from dateutil import parser

my_api_key = "***************************"
my_cse_id = "***********************" #The google custom search-engine-ID you created
client = OpenAI()

def get_nfl_week(date_string):
    """
    Determines the NFL week and year for a given date_string.
    
    Args:
    date_string (str): The date in 'YYYYMMDD' format.
    
    Returns:
    tuple: A tuple containing 'Game Day: YYYY Week X' and 'YYYY Week X'.
    """
    # Convert date_string to a datetime object
    date = datetime.strptime(date_string, '%Y%m%d')
    
    # Define the start dates of NFL seasons from 2019 to 2024
    nfl_start_dates = {
        2019: datetime(2019, 9, 5),
        2020: datetime(2020, 9, 10),
        2021: datetime(2021, 9, 9),
        2022: datetime(2022, 9, 8),
        2023: datetime(2023, 9, 7),
        2024: datetime(2024, 9, 5),
    }
    
    # Determine the year of the NFL season
    season_year = None
    for year, start_date in sorted(nfl_start_dates.items()):
        if date >= start_date:
            season_year = year
        else:
            break
    
    # Calculate the NFL week number
    if season_year is not None:
        season_start = nfl_start_dates[season_year]
        delta = date - season_start
        week_number = delta.days // 7 + 1
        nfl_week_str = f'Week {week_number}'
    else:
        raise ValueError("Date is out of range for the available NFL seasons.")
    
    return nfl_week_str

def is_article_before_date(url, cutoff_date):
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    date_tag = soup.find('meta', {'property': 'article:published_time'})
    if date_tag:
        try:
            # Parse the article date, which may include timezone information
            article_date = parser.parse(date_tag['content'])

            # Make the cutoff date aware of timezone if it is naive
            if article_date.tzinfo is not None and cutoff_date.tzinfo is None:
                cutoff_date = cutoff_date.replace(tzinfo=article_date.tzinfo)

            # Compare dates
            return article_date < cutoff_date
        except Exception as e:
            print(f"Error parsing date from URL {url}: {e}")
            return False
    return False

def google_search(date_string, api_key=my_api_key, cse_id=my_cse_id, max_results=50, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Calculate the contest date and the 5 days prior
    contest_date = datetime.strptime(date_string, "%Y%m%d")
    start_date = (contest_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = (contest_date - timedelta(days=1)).strftime('%Y-%m-%d')
    search_term = f"NFL DFS {date_string[:4]} {get_nfl_week(date_string)} Players"

    results = []
    num = 10  # Maximum number of results per request

    try:
        for start in range(1, max_results, num):
            res = service.cse().list(
                q=search_term,
                cx=cse_id,
                num=num,
                start=start,
                dateRestrict=f"before:{date_string}",
                **kwargs
            ).execute()
            
            # Append current batch of results
            items = res.get('items', [])
            results.extend(items)
            
            # Check if we've reached the end of the results
            if len(items) < num:
                break

        return results
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

def google_search_waiver_wire(date_string, api_key=my_api_key, cse_id=my_cse_id, max_results=50, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Calculate the contest date and the 5 days prior
    contest_date = datetime.strptime(date_string, "%Y%m%d")
    start_date = (contest_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = (contest_date - timedelta(days=1)).strftime('%Y-%m-%d')
    search_term = f"NFL fantasy football waiver wire {date_string[:4]} {get_nfl_week(date_string)}"

    results = []
    num = 10  # Maximum number of results per request

    try:
        for start in range(1, max_results, num):
            res = service.cse().list(
                q=search_term,
                cx=cse_id,
                num=num,
                start=start,
                **kwargs
            ).execute()
            
            # Append current batch of results
            items = res.get('items', [])
            results.extend(items)
            
            # Check if we've reached the end of the results
            if len(items) < num:
                break

        return results
    except Exception as e:
        print(f"Error during Google search (waiver wire): {e}")
        return []


def truncate_text(text, max_tokens=2000):
    # This function will truncate text to approximately the number of specified tokens
    words = text.split()
    truncated_words = words[:max_tokens]
    return ' '.join(truncated_words)

def extract_text_from_url(url, timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            response.encoding = 'utf-8'  # Ensure the encoding is set to UTF-8
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and remove extra whitespace
            text = soup.get_text(separator="\n")
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
            text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
            
            return text.strip()
        else:
            return None
    except requests.Timeout:
        print(f"Timeout fetching URL {url}")
        return None
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def filter_relevant_articles(text, date_string, url):
    """
    Uses OpenAI to determine if the article is relevant to DraftKings player recommendations.
    """
    contest_date = datetime.strptime(date_string, '%Y%m%d')

    # Check if the article was published before the contest date
    if not is_article_before_date(url, contest_date):
        return False

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a daily fantasy sports player creating a lineup for an upcoming contest."},
                {"role": "user", "content": f"Is this article providing recommendations for upcoming regular season NFL fantasy sports contests in {date_string[:4]} {get_nfl_week(date_string)}? I need you to be extremely confident that this information is specifically for regular season {date_string[:4]} {get_nfl_week(date_string)} and not another year or preseason. Answer 'Yes' or if you are not confident answer 'No'. Text: {truncate_text(text)}"}
            ]
        )
        response_text = completion.choices[0].message.content
        #print (response_text)
        return response_text.strip().lower().startswith('yes')
    except Exception as e:
        print(f"Error with OpenAI API (filtering): {e}")
        return False

def analyze_article_with_openai(text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a daily fantasy sports player creating a lineup for an upcoming contest."},
                {"role": "user", "content": f"Identify which players are mentioned in this text and rate their fantasy value from 0 (poor pick) to 10 (excellent pick). Return the response strictly in the format '[Player Name]: [Rating]' with each player and rating on a new line. Do not include any additional text. Text: {truncate_text(text)}"}
            ]
        )
        response_text = completion.choices[0].message.content.strip()
        
        # Split the response text into lines and parse player data
        player_data = []
        for line in response_text.split('\n'):
            if line.strip():
                player, rating = line.split(':')
                player_data.append({'Name': player.strip(), 'Score': int(rating.strip())})
        
        return player_data
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return None



def search_and_analyze_draftkings(date_string, path_to_proj):
    # Step 1: Google Search
    search_results = google_search(date_string)

    # Step 2: Extract and process text from URLs
    articles_text = []
    relevant_articles_text = []
    for result in search_results[:20]:  # Limit to top 20 results
        url = result.get('link')
        article_text = extract_text_from_url(url)
        if article_text:
            articles_text.append(article_text)
            if filter_relevant_articles(article_text, date_string, url):
                relevant_articles_text.append(article_text)

    # Step 3: Analyze content with OpenAI
    player_ratings = []
    for text in relevant_articles_text:
        analysis_result = analyze_article_with_openai(text)
        if analysis_result:
            # Directly use the analysis result (list of dictionaries) returned by OpenAI
            player_ratings.extend(analysis_result)  # Add the list of players and ratings to player_ratings
    
    # Convert player ratings to DataFrame
    players_df = pd.DataFrame(player_ratings)

    # Step 4: Remap names and aggregate scores
    if players_df.empty or 'Name' not in players_df.columns:
        return pd.DataFrame(columns=['Final', 'ExpertAverageScore', 'ExpertNumMentions'])
    
    players_df = players_df.dropna(subset=['Name'])
    players_df['Final'] = NameMatching.remap_player_names(players_df['Name'], path_to_proj)
    aggregated_df = players_df.groupby('Final').agg({'Score': ['mean', 'count']}).reset_index()
    aggregated_df.columns = ['Final', 'ExpertAverageScore', 'ExpertNumMentions']
    
    return aggregated_df

def search_and_analyze_waiver_wire(date_string, path_to_proj):
    # Step 1: Google Search
    search_results = google_search_waiver_wire(date_string)

    # Step 2: Extract and process text from URLs
    articles_text = []
    relevant_articles_text = []
    for result in search_results[:20]:  # Limit to top 20 results
        url = result.get('link')
        article_text = extract_text_from_url(url)
        if article_text:
            articles_text.append(article_text)
            if filter_relevant_articles(article_text, date_string, url):
                relevant_articles_text.append(article_text)

    # Step 3: Analyze content with OpenAI
    player_ratings = []
    for text in relevant_articles_text:
        analysis_result = analyze_article_with_openai(text)
        if analysis_result:
            # Directly use the analysis result (list of dictionaries) returned by OpenAI
            player_ratings.extend(analysis_result)  # Add the list of players and ratings to player_ratings
    
    # Convert player ratings to DataFrame
    players_df = pd.DataFrame(player_ratings)

    # Step 4: Remap names and aggregate scores
    if players_df.empty or 'Name' not in players_df.columns:
        return pd.DataFrame(columns=['Final', 'WaiverAverageScore', 'WaiverNumMentions'])
    
    players_df = players_df.dropna(subset=['Name'])
    players_df['Final'] = NameMatching.remap_player_names(players_df['Name'], path_to_proj)
    aggregated_df = players_df.groupby('Final').agg({'Score': ['mean', 'count']}).reset_index()
    aggregated_df.columns = ['Final', 'WaiverAverageScore', 'WaiverNumMentions']
    
    return aggregated_df

def run_expert_analysis(date_string, path_to_proj, player_salaries):
    """
    Perform a search and analysis of DraftKings picks for a given date, save the results,
    and merge the aggregated results with player salaries.

    Args:
    - date_string (str): The date for the DraftKings contest in YYYYMMDD format.
    - path_to_proj (str): The base path to the project directory.
    - player_salaries (pd.DataFrame): DataFrame containing player salary data.

    Returns:
    - pd.DataFrame: The updated player_salaries DataFrame with merged expert analysis data.
    """

    # Step 1: Search and analyze DraftKings picks
    aggregated_df = search_and_analyze_draftkings(date_string, path_to_proj)
    
    # Step 2: Search and analyze waiver wire picks
    waiver_aggregated_df = search_and_analyze_waiver_wire(date_string, path_to_proj)

    # Step 3: Save the aggregated results to CSV
    experts_output_path = os.path.join(path_to_proj, f"Data/Experts/Experts{date_string}.csv")
    aggregated_df.to_csv(experts_output_path, index=False)
    
    waiver_output_path = os.path.join(path_to_proj, f"Data/Experts/WaiverWire{date_string}.csv")
    waiver_aggregated_df.to_csv(waiver_output_path, index=False)

    # Step 4: Merge aggregated results with player_salaries
    merged_df = player_salaries.merge(aggregated_df, left_on='Name', right_on='Final', how='left')
    merged_df = merged_df.merge(waiver_aggregated_df, left_on='Name', right_on='Final', how='left')
    
    # Clean up the merged DataFrame: Drop the 'Final' columns from aggregated_df
    merged_df.drop(columns=['Final_x', 'Final_y'], inplace=True)
    merged_df['ExpertAverageScore'].fillna(5, inplace=True)  # Fill missing AverageScore with 5
    merged_df['ExpertNumMentions'].fillna(0, inplace=True)   # Fill missing NumMentions with 0
    merged_df['WaiverAverageScore'].fillna(5, inplace=True)  # Fill missing WaiverAverageScore with 5
    merged_df['WaiverNumMentions'].fillna(0, inplace=True)   # Fill missing WaiverNumMentions with 0

    # Optional: Print the number of matches
    #print(f"Number of matched players: {merged_df['Score'].notna().sum()}")

    return merged_df