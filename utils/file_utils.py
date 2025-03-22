import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure that a directory exists, creating it if necessary"""
    os.makedirs(directory_path, exist_ok=True)

def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file"""
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str, default=None) -> Any:
    """Load data from a JSON file, returning default if the file doesn't exist"""
    if not os.path.exists(file_path):
        return default
    
    with open(file_path, 'r') as f:
        return json.load(f)

def save_match_analysis(match_id: str, analysis: Dict, data_dir: str = 'data') -> str:
    """Save match analysis to a JSON file"""
    matches_dir = os.path.join(data_dir, 'matches')
    ensure_directory_exists(matches_dir)
    
    # Generate filename with date for easier sorting
    date_str = datetime.now().strftime('%Y%m%d')
    file_path = os.path.join(matches_dir, f"{date_str}_{match_id}.json")
    
    save_json(analysis, file_path)
    return file_path

def load_match_analysis(match_id: str, data_dir: str = 'data') -> Optional[Dict]:
    """Load match analysis from a JSON file"""
    matches_dir = os.path.join(data_dir, 'matches')
    
    # Try to find the file by match_id (without knowing the exact date)
    for filename in os.listdir(matches_dir):
        if match_id in filename and filename.endswith('.json'):
            file_path = os.path.join(matches_dir, filename)
            return load_json(file_path)
    
    return None

def load_all_match_analyses(data_dir: str = 'data') -> List[Dict]:
    """Load all match analyses from the data directory"""
    matches_dir = os.path.join(data_dir, 'matches')
    ensure_directory_exists(matches_dir)
    
    analyses = []
    for filename in os.listdir(matches_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(matches_dir, filename)
            analysis = load_json(file_path)
            if analysis:
                analyses.append(analysis)
    
    return analyses

def export_to_csv(data: List[Dict], file_path: str) -> None:
    """Export data to a CSV file"""
    df = pd.DataFrame(data)
    directory = os.path.dirname(file_path)
    ensure_directory_exists(directory)
    df.to_csv(file_path, index=False)

def format_odds(odds: float, format_type: str = 'decimal') -> str:
    """Format odds in different formats (decimal, fractional, american)"""
    if format_type == 'decimal':
        return f"{odds:.2f}"
    elif format_type == 'fractional':
        # Convert decimal to fractional (approximate)
        decimal_part = odds - 1.0
        if decimal_part == 0:
            return "0/1"
        
        # Convert to fraction with reasonable denominator
        for denominator in [2, 3, 4, 5, 6, 8, 10, 20, 25, 50, 100]:
            numerator = round(decimal_part * denominator)
            if abs(decimal_part - (numerator / denominator)) < 0.01:
                return f"{numerator}/{denominator}"
        
        # Fallback for more complex fractions
        numerator = round(decimal_part * 100)
        denominator = 100
        return f"{numerator}/{denominator}"
    elif format_type == 'american':
        if odds >= 2.0:
            return f"+{int((odds - 1) * 100)}"
        else:
            return f"-{int(100 / (odds - 1))}"
    else:
        return str(odds)