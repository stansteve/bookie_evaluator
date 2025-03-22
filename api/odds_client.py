import os
import json
import requests
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class OddsAPI:
    """Client for fetching odds data from The Odds API"""
    
    def __init__(self):
        self.api_key = os.getenv('ODDS_API_KEY')
        self.base_url = 'https://api.the-odds-api.com/v4'
        self.cache_dir = os.path.join(os.getenv('DATA_DIRECTORY', 'data'), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_sports(self):
        """Get a list of available sports"""
        endpoint = f"{self.base_url}/sports"
        params = {'apiKey': self.api_key}
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching sports: {response.status_code}")
            return []
        
        return response.json()
    
    def get_odds(self, sport="soccer", regions="eu", markets="h2h,spreads,totals", date_format="iso"):
        """Get odds for upcoming matches for a specific sport"""
        cache_file = os.path.join(self.cache_dir, f"{sport}_odds.json")
        
        # Check if we have cached data that's not expired
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                cache_time = data.get('timestamp', 0)
                
                # If cache is not expired, return cached data
                if time.time() - cache_time < int(os.getenv('CACHE_EXPIRY', 3600)):
                    return data.get('odds', [])
        
        # Fetch new data
        endpoint = f"{self.base_url}/sports/{sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
            'dateFormat': date_format
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching odds: {response.status_code}")
            return []
        
        odds_data = response.json()
        
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'odds': odds_data
            }, f)
        
        return odds_data
    
    def get_historical_odds(self, sport="soccer", days_ago=30):
        """Get historical odds for past matches"""
        # In a real implementation, you might need to use a different API or data source
        # This is a placeholder for demonstration
        
        # Simulate historical data by creating a structured DataFrame
        past_date = datetime.now() - timedelta(days=days_ago)
        
        # This would normally come from an API or database
        historical_data = []
        
        return historical_data


class FootballDataAPI:
    """Client for fetching football match data from Football-Data.org"""
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_API_KEY')
        self.base_url = 'https://api.football-data.org/v4'
        self.cache_dir = os.path.join(os.getenv('DATA_DIRECTORY', 'data'), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_competitions(self):
        """Get a list of available competitions"""
        endpoint = f"{self.base_url}/competitions"
        headers = {'X-Auth-Token': self.api_key}
        
        response = requests.get(endpoint, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching competitions: {response.status_code}")
            return []
        
        return response.json().get('competitions', [])
    
    def get_matches(self, competition_id='PL', status='SCHEDULED'):
        """Get matches for a specific competition"""
        cache_file = os.path.join(self.cache_dir, f"{competition_id}_matches.json")
        
        # Check if we have cached data that's not expired
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                cache_time = data.get('timestamp', 0)
                
                # If cache is not expired, return cached data
                if time.time() - cache_time < int(os.getenv('CACHE_EXPIRY', 3600)):
                    return data.get('matches', [])
        
        # Fetch new data
        endpoint = f"{self.base_url}/competitions/{competition_id}/matches"
        headers = {'X-Auth-Token': self.api_key}
        params = {'status': status}
        
        response = requests.get(endpoint, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching matches: {response.status_code}")
            return []
        
        matches_data = response.json().get('matches', [])
        
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'matches': matches_data
            }, f)
        
        return matches_data
    
    def get_team_matches(self, team_id, status='FINISHED', limit=10):
        """Get recent matches for a specific team"""
        endpoint = f"{self.base_url}/teams/{team_id}/matches"
        headers = {'X-Auth-Token': self.api_key}
        params = {'status': status, 'limit': limit}
        
        response = requests.get(endpoint, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching team matches: {response.status_code}")
            return []
        
        return response.json().get('matches', [])