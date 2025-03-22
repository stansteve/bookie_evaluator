#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.odds_client import OddsAPI, FootballDataAPI
from models.odds_analyzer import OddsAnalyzer
from ml import FootballMatchPredictor
from utils.file_utils import (
    save_match_analysis, 
    load_match_analysis, 
    load_all_match_analyses,
    export_to_csv,
    format_odds
)

class BookieEvaluator:
    """Main application for evaluating outcomes where bookmakers are consistently accurate"""
    
    def __init__(self):
        self.odds_api = OddsAPI()
        self.football_api = FootballDataAPI()
        self.analyzer = OddsAnalyzer(data_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.ml_predictor = FootballMatchPredictor(
            data_dir=self.data_dir,
            model_dir='ml/models'
        )
    
    def fetch_upcoming_matches(self, sport="soccer", competition="PL"):
        """Fetch upcoming matches for analysis"""
        print(f"{Fore.CYAN}Fetching upcoming {sport} matches for {competition}...{Style.RESET_ALL}")
        
        # Get matches from Football-Data API
        matches = self.football_api.get_matches(competition_id=competition, status='SCHEDULED')
        
        # Get odds from The Odds API
        odds_data = self.odds_api.get_odds(sport=sport)
        
        print(f"{Fore.GREEN}Found {len(matches)} upcoming matches and odds from {len(odds_data)} events.{Style.RESET_ALL}")
        
        # Return both data sources for further processing
        return matches, odds_data
    
    def analyze_match(self, match_id, match_data, odds_data):
        """Analyze odds for a specific match"""
        # Find match in both data sources
        match = next((m for m in match_data if str(m.get('id')) == str(match_id)), None)
        if not match:
            print(f"{Fore.RED}Match ID {match_id} not found in the data.{Style.RESET_ALL}")
            return None
        
        home_team = match.get('homeTeam', {}).get('name', '')
        away_team = match.get('awayTeam', {}).get('name', '')
        match_date = match.get('utcDate', '')
        
        print(f"{Fore.CYAN}Analyzing match: {home_team} vs {away_team} on {match_date}{Style.RESET_ALL}")
        
        # Find corresponding odds data
        # Note: In a real implementation, you would need to match events across APIs
        # This is simplified for demonstration
        odds_match = next((
            o for o in odds_data if 
            (home_team.lower() in o.get('home_team', '').lower() or 
             o.get('home_team', '').lower() in home_team.lower()) and
            (away_team.lower() in o.get('away_team', '').lower() or 
             o.get('away_team', '').lower() in away_team.lower())
        ), None)
        
        if not odds_match:
            print(f"{Fore.YELLOW}Could not find odds data for {home_team} vs {away_team}.{Style.RESET_ALL}")
            return None
        
        # Analyze the odds
        analysis = self.analyzer.analyze_match_odds(odds_match)
        
        # Add match details
        analysis['match_date'] = match_date
        analysis['competition'] = match.get('competition', {}).get('name', '')
        
        # Save the analysis
        save_path = save_match_analysis(match_id, analysis, self.data_dir)
        print(f"{Fore.GREEN}Match analysis saved to {save_path}{Style.RESET_ALL}")
        
        return analysis
    
    def display_analysis(self, analysis):
        """Display match analysis in a formatted table"""
        if not analysis:
            return
        
        print(f"\n{Fore.CYAN}Match: {analysis['home_team']} vs {analysis['away_team']}{Style.RESET_ALL}")
        print(f"Date: {analysis.get('match_date', 'Unknown')}")
        print(f"Competition: {analysis.get('competition', 'Unknown')}\n")
        
        # Prepare data for tabulation
        table_data = []
        for bookie in analysis.get('bookmakers', []):
            home_odds = bookie.get('home_odds', 0)
            draw_odds = bookie.get('draw_odds', 0)
            away_odds = bookie.get('away_odds', 0)
            
            home_prob = bookie.get('implied_home_prob', 0) * 100
            draw_prob = bookie.get('implied_draw_prob', 0) * 100
            away_prob = bookie.get('implied_away_prob', 0) * 100
            
            margin = bookie.get('margin', 0) * 100
            
            table_data.append([
                bookie.get('name', 'Unknown'),
                f"{format_odds(home_odds)} ({home_prob:.1f}%)",
                f"{format_odds(draw_odds)} ({draw_prob:.1f}%)",
                f"{format_odds(away_odds)} ({away_prob:.1f}%)",
                f"{margin:.1f}%"
            ])
        
        # Sort by margin (ascending)
        table_data.sort(key=lambda x: float(x[4].replace('%', '')))
        
        # Display the table
        headers = ['Bookmaker', 'Home (prob)', 'Draw (prob)', 'Away (prob)', 'Margin']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def identify_value_bets(self, analysis, confidence_threshold=0.05):
        """Identify and display potential value bets"""
        if not analysis:
            return
        
        value_bets = self.analyzer.identify_value_bets(analysis, confidence_threshold)
        
        if not value_bets:
            print(f"\n{Fore.YELLOW}No significant value bets identified for this match.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Potential Value Bets:{Style.RESET_ALL}")
        
        # Prepare data for tabulation
        table_data = []
        for bet in value_bets:
            bet_type = bet.get('type', '').upper()
            bookmaker = bet.get('bookmaker', '')
            odds = bet.get('odds', 0)
            implied_prob = bet.get('implied_prob', 0) * 100
            consensus_prob = bet.get('consensus_prob', 0) * 100
            diff = bet.get('diff', 0) * 100
            ev = bet.get('expected_value', 0)
            
            table_data.append([
                bet_type,
                bookmaker,
                f"{format_odds(odds)}",
                f"{implied_prob:.1f}%",
                f"{consensus_prob:.1f}%",
                f"{diff:.1f}%",
                f"{ev:.2f}"
            ])
        
        # Display the table
        headers = ['Bet Type', 'Bookmaker', 'Odds', 'Bookmaker Prob', 'Consensus Prob', 'Difference', 'Expected Value']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def display_bookmaker_rankings(self, min_matches=10):
        """Display bookmaker accuracy rankings"""
        rankings = self.analyzer.get_bookmaker_rankings(min_matches=min_matches)
        
        if not rankings:
            print(f"\n{Fore.YELLOW}No bookmaker rankings available yet. Need more historical data.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Bookmaker Accuracy Rankings (min. {min_matches} matches):{Style.RESET_ALL}")
        
        # Prepare data for tabulation
        table_data = []
        for i, bookie in enumerate(rankings):
            name = bookie.get('name', 'Unknown')
            matches = bookie.get('matches', 0)
            accuracy = bookie.get('accuracy', 0) * 100
            brier = bookie.get('brier_score', 0)
            log_loss = bookie.get('log_loss', 0)
            margin = bookie.get('avg_margin', 0) * 100
            
            table_data.append([
                i + 1,
                name,
                matches,
                f"{accuracy:.1f}%",
                f"{brier:.3f}",
                f"{log_loss:.3f}",
                f"{margin:.1f}%"
            ])
        
        # Display the table
        headers = ['Rank', 'Bookmaker', 'Matches', 'Accuracy', 'Brier Score', 'Log Loss', 'Avg Margin']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def record_match_result(self, match_id, result):
        """Record the actual result of a match and update outcome evaluations"""
        # Load the pre-match analysis
        analysis = load_match_analysis(match_id, self.data_dir)
        
        if not analysis:
            print(f"{Fore.RED}Could not find pre-match analysis for match ID {match_id}.{Style.RESET_ALL}")
            return False
        
        # Validate the result
        if result not in ['home', 'draw', 'away']:
            print(f"{Fore.RED}Invalid result '{result}'. Must be one of: home, draw, away{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}Recording match result for {analysis['home_team']} vs {analysis['away_team']}: {result.upper()}{Style.RESET_ALL}")
        
        # Evaluate bookmaker accuracy for this match (legacy)
        self.analyzer.evaluate_bookmaker_accuracy(match_id, result, analysis)
        
        # Evaluate outcome reliability for this match (new approach)
        self.analyzer.evaluate_outcome_reliability(match_id, result, analysis)
        
        print(f"{Fore.GREEN}Outcome reliability data updated.{Style.RESET_ALL}")
        return True
    
    def display_outcome_reliability_stats(self):
        """Display statistics on outcome prediction reliability"""
        stats = self.analyzer.get_outcome_reliability_stats()
        
        if not stats:
            print(f"\n{Fore.YELLOW}No outcome reliability statistics available yet. Need more historical data.{Style.RESET_ALL}")
            return
        
        # Display overall stats
        print(f"\n{Fore.CYAN}Overall Outcome Prediction Accuracy:{Style.RESET_ALL}")
        
        overall_table = []
        for outcome in ['home', 'draw', 'away']:
            correct = stats['overall'][outcome]['correct']
            total = stats['overall'][outcome]['total']
            accuracy = stats['overall'][outcome]['accuracy'] * 100
            
            overall_table.append([
                outcome.upper(),
                correct,
                total,
                f"{accuracy:.1f}%"
            ])
        
        print(tabulate(overall_table, headers=['Outcome', 'Correct', 'Total', 'Accuracy'], tablefmt='grid'))
        
        # Display stats by confidence bands
        print(f"\n{Fore.CYAN}Accuracy by Confidence Level:{Style.RESET_ALL}")
        
        confidence_table = []
        for band in ['very_high', 'high', 'medium', 'low', 'very_low']:
            if band not in stats['by_confidence']:
                continue
                
            for outcome in ['home', 'draw', 'away']:
                outcome_stats = stats['by_confidence'][band].get(outcome, {})
                correct = outcome_stats.get('correct', 0)
                total = outcome_stats.get('total', 0)
                
                if total == 0:
                    continue
                    
                accuracy = outcome_stats.get('accuracy', 0) * 100
                
                confidence_table.append([
                    band.replace('_', ' ').title(),
                    outcome.upper(),
                    correct,
                    total,
                    f"{accuracy:.1f}%"
                ])
        
        print(tabulate(confidence_table, 
                       headers=['Confidence', 'Outcome', 'Correct', 'Total', 'Accuracy'], 
                       tablefmt='grid'))
        
        # Display stats by consensus levels
        print(f"\n{Fore.CYAN}Accuracy by Consensus Level:{Style.RESET_ALL}")
        
        consensus_table = []
        for level in ['strong', 'moderate', 'weak']:
            if level not in stats['by_consensus']:
                continue
                
            for outcome in ['home', 'draw', 'away']:
                outcome_stats = stats['by_consensus'][level].get(outcome, {})
                correct = outcome_stats.get('correct', 0)
                total = outcome_stats.get('total', 0)
                
                if total == 0:
                    continue
                    
                accuracy = outcome_stats.get('accuracy', 0) * 100
                
                consensus_table.append([
                    level.title(),
                    outcome.upper(),
                    correct,
                    total,
                    f"{accuracy:.1f}%"
                ])
        
        print(tabulate(consensus_table, 
                       headers=['Consensus', 'Outcome', 'Correct', 'Total', 'Accuracy'], 
                       tablefmt='grid'))
    
    def identify_reliable_bets(self, match_analysis, min_accuracy=0.7, min_matches=5):
        """Identify and display historically reliable betting opportunities"""
        if not match_analysis:
            return
        
        reliable_outcomes = self.analyzer.identify_reliable_outcomes(
            match_analysis, min_accuracy=min_accuracy, min_matches=min_matches
        )
        
        if not reliable_outcomes:
            print(f"\n{Fore.YELLOW}No reliable betting opportunities identified for this match based on historical patterns.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}This may be due to insufficient historical data or no patterns meeting the minimum criteria.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}Historically Reliable Betting Opportunities:{Style.RESET_ALL}")
        
        # Prepare data for tabulation
        table_data = []
        for outcome, data in reliable_outcomes.items():
            outcome_type = outcome.upper()
            prob = data.get('implied_probability', 0) * 100
            odds = data.get('average_odds', 0)
            conf_band = data.get('confidence_band', '').replace('_', ' ').title()
            cons_level = data.get('consensus_level', '').title()
            hist_acc = data.get('historical_accuracy', 0) * 100
            hist_samples = data.get('historical_samples', 0)
            exp_value = data.get('expected_value', 0)
            
            table_data.append([
                outcome_type,
                f"{prob:.1f}%",
                f"{format_odds(odds)}",
                conf_band,
                cons_level,
                f"{hist_acc:.1f}%",
                hist_samples,
                f"{exp_value:.2f}"
            ])
        
        # Sort by historical accuracy (descending)
        table_data.sort(key=lambda x: float(x[5].replace('%', '')), reverse=True)
        
        # Display the table
        headers = ['Outcome', 'Probability', 'Avg Odds', 'Confidence', 'Consensus', 'Historical Accuracy', 'Samples', 'Expected Value']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def train_ml_models(self, test_size=0.2, random_state=42):
        """Train machine learning models using historical match data"""
        print(f"{Fore.CYAN}Loading historical match data for ML training...{Style.RESET_ALL}")
        
        # Load all match analyses with results
        all_matches = []
        
        # First, get completed matches from outcome performance data
        outcome_matches = self.analyzer.outcome_performance.get('matches', [])
        
        # Also load raw match analyses
        raw_matches = load_all_match_analyses(self.data_dir)
        
        # Combine and ensure we have actual results
        for match in raw_matches:
            match_id = match.get('match_id', '')
            outcome_match = next((m for m in outcome_matches if m.get('match_id') == match_id), None)
            
            if outcome_match:
                # Add the actual result to the match analysis
                match['actual_result'] = outcome_match.get('actual_result')
                all_matches.append(match)
        
        if not all_matches:
            print(f"{Fore.RED}No historical match data with results found. Record some match results first.{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.GREEN}Found {len(all_matches)} historical matches with results for training.{Style.RESET_ALL}")
        
        # Prepare training data
        X, y = self.ml_predictor.prepare_training_data(all_matches)
        
        if X.empty or y.empty:
            print(f"{Fore.RED}Failed to extract features from historical matches.{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}Training machine learning models...{Style.RESET_ALL}")
        results = self.ml_predictor.train_models(X, y, test_size=test_size, random_state=random_state)
        
        if 'error' in results:
            print(f"{Fore.RED}Error training models: {results['error']}{Style.RESET_ALL}")
            return False
        
        # Display results
        print(f"{Fore.GREEN}Model training complete! Results:{Style.RESET_ALL}")
        
        table_data = []
        for model_name, metrics in results.items():
            accuracy = metrics.get('accuracy', 0) * 100
            cv_mean = metrics.get('cross_val_mean', 0) * 100
            cv_std = metrics.get('cross_val_std', 0) * 100
            brier = metrics.get('brier_score', 0)
            
            table_data.append([
                model_name.replace('_', ' ').title(),
                f"{accuracy:.1f}%",
                f"{cv_mean:.1f}% (±{cv_std:.1f}%)",
                f"{brier:.4f}"
            ])
        
        # Sort by accuracy (descending)
        table_data.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)
        
        headers = ['Model', 'Test Accuracy', 'Cross-Val Accuracy', 'Brier Score']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Display feature importance
        try:
            self.ml_predictor.plot_feature_importance(model_name='ensemble', top_n=15)
        except Exception as e:
            print(f"{Fore.YELLOW}Could not plot feature importance: {str(e)}{Style.RESET_ALL}")
        
        return True
    
    def hyperparameter_tuning(self, model_type='xgboost', n_trials=50, test_size=0.2, random_state=42):
        """Perform hyperparameter tuning for a specific model type"""
        print(f"{Fore.CYAN}Loading historical match data for hyperparameter tuning...{Style.RESET_ALL}")
        
        # Load all match analyses with results (similar to train_ml_models)
        all_matches = []
        outcome_matches = self.analyzer.outcome_performance.get('matches', [])
        raw_matches = load_all_match_analyses(self.data_dir)
        
        for match in raw_matches:
            match_id = match.get('match_id', '')
            outcome_match = next((m for m in outcome_matches if m.get('match_id') == match_id), None)
            
            if outcome_match:
                match['actual_result'] = outcome_match.get('actual_result')
                all_matches.append(match)
        
        if not all_matches:
            print(f"{Fore.RED}No historical match data with results found. Record some match results first.{Style.RESET_ALL}")
            return False
        
        # Prepare training data
        X, y = self.ml_predictor.prepare_training_data(all_matches)
        
        if X.empty or y.empty:
            print(f"{Fore.RED}Failed to extract features from historical matches.{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}Starting hyperparameter tuning for {model_type} model with {n_trials} trials...{Style.RESET_ALL}")
        results = self.ml_predictor.hypertune_model(
            X, y, model_type=model_type, n_trials=n_trials, test_size=test_size, random_state=random_state
        )
        
        if 'error' in results:
            print(f"{Fore.RED}Error during hyperparameter tuning: {results['error']}{Style.RESET_ALL}")
            return False
        
        # Display results
        print(f"{Fore.GREEN}Hyperparameter tuning complete! Results:{Style.RESET_ALL}")
        
        accuracy = results.get('test_accuracy', 0) * 100
        best_cv_score = results.get('best_cv_score', 0) * 100
        brier = results.get('brier_score', 0)
        
        print(f"Best parameters: {results.get('best_params', {})}")
        print(f"Test accuracy: {accuracy:.2f}%")
        print(f"Best CV score: {best_cv_score:.2f}%")
        print(f"Brier score: {brier:.4f}")
        
        # Display classification report
        report = results.get('classification_report', {})
        if report:
            report_df = pd.DataFrame(report).transpose()
            print("\nClassification Report:")
            print(tabulate(report_df, headers='keys', tablefmt='grid'))
        
        # Generate SHAP analysis
        try:
            print(f"{Fore.CYAN}Generating SHAP analysis for model interpretability...{Style.RESET_ALL}")
            self.ml_predictor.save_shap_analysis(X, model_name=f'optimized_{model_type}')
        except Exception as e:
            print(f"{Fore.YELLOW}Could not generate SHAP analysis: {str(e)}{Style.RESET_ALL}")
        
        return True
    
    def predict_with_ml(self, match_analysis, model_name='ensemble', bookmaker_weight=0.5):
        """Predict match outcome using machine learning models"""
        if not match_analysis:
            return
        
        print(f"{Fore.CYAN}Predicting match outcome with ML model: {model_name}{Style.RESET_ALL}")
        
        # Get pure ML prediction
        ml_prediction = self.ml_predictor.predict_match(match_analysis, model_name=model_name)
        
        if 'error' in ml_prediction:
            print(f"{Fore.RED}ML prediction error: {ml_prediction['error']}{Style.RESET_ALL}")
            return
        
        # Get combined prediction (ML + bookmakers)
        combined_prediction = self.ml_predictor.combine_predictions(
            match_analysis, bookmaker_weight=bookmaker_weight
        )
        
        # Display ML prediction
        print(f"\n{Fore.GREEN}Machine Learning Prediction:{Style.RESET_ALL}")
        outcome = ml_prediction.get('predicted_outcome', '').upper()
        confidence = ml_prediction.get('confidence', 0) * 100
        
        print(f"Predicted outcome: {outcome} (Confidence: {confidence:.1f}%)")
        print(f"Home win: {ml_prediction.get('home_prob', 0) * 100:.1f}%")
        print(f"Draw: {ml_prediction.get('draw_prob', 0) * 100:.1f}%")
        print(f"Away win: {ml_prediction.get('away_prob', 0) * 100:.1f}%")
        print(f"Model used: {ml_prediction.get('model', '')}")
        
        # Display combined prediction
        print(f"\n{Fore.GREEN}Combined Prediction (ML + Bookmakers):{Style.RESET_ALL}")
        combined_outcome = combined_prediction.get('predicted_outcome', '').upper()
        combined_confidence = combined_prediction.get('confidence', 0) * 100
        
        print(f"Predicted outcome: {combined_outcome} (Confidence: {combined_confidence:.1f}%)")
        print(f"Home win: {combined_prediction.get('home_prob', 0) * 100:.1f}%")
        print(f"Draw: {combined_prediction.get('draw_prob', 0) * 100:.1f}%")
        print(f"Away win: {combined_prediction.get('away_prob', 0) * 100:.1f}%")
        print(f"Bookmaker weight: {bookmaker_weight * 100:.0f}%")
        print(f"ML weight: {(1 - bookmaker_weight) * 100:.0f}%")
        
        return combined_prediction


def main():
    parser = argparse.ArgumentParser(description='Bookie Evaluator - Identify reliable betting opportunities')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Fetch matches command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch upcoming matches')
    fetch_parser.add_argument('--sport', '-s', default='soccer', help='Sport to fetch matches for')
    fetch_parser.add_argument('--competition', '-c', default='PL', help='Competition ID (e.g., PL for English Premier League)')
    
    # Analyze match command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze odds for a specific match')
    analyze_parser.add_argument('match_id', help='ID of the match to analyze')
    
    # Record result command
    record_parser = subparsers.add_parser('record', help='Record the result of a match')
    record_parser.add_argument('match_id', help='ID of the match')
    record_parser.add_argument('result', choices=['home', 'draw', 'away'], help='Match result (home, draw, away)')
    
    # Reliability stats command
    stats_parser = subparsers.add_parser('stats', help='Display outcome reliability statistics')
    
    # Find reliable bets command
    reliable_parser = subparsers.add_parser('reliable', help='Find historically reliable betting opportunities for a match')
    reliable_parser.add_argument('match_id', help='ID of the match to analyze')
    reliable_parser.add_argument('--min-accuracy', '-a', type=float, default=0.7, help='Minimum historical accuracy (0.0-1.0)')
    reliable_parser.add_argument('--min-matches', '-m', type=int, default=5, help='Minimum number of historical matches')
    
    # Train ML models command
    train_ml_parser = subparsers.add_parser('train-ml', help='Train machine learning models on historical data')
    train_ml_parser.add_argument('--test-size', '-t', type=float, default=0.2, help='Proportion of data to use for testing')
    train_ml_parser.add_argument('--random-state', '-r', type=int, default=42, help='Random seed for reproducibility')
    
    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser('tune', help='Perform hyperparameter tuning for ML models')
    tune_parser.add_argument('--model', '-m', choices=['xgboost', 'lightgbm', 'random_forest', 'logistic_regression'], 
                            default='xgboost', help='Model type to tune')
    tune_parser.add_argument('--trials', '-n', type=int, default=50, help='Number of optimization trials')
    tune_parser.add_argument('--test-size', '-t', type=float, default=0.2, help='Proportion of data to use for testing')
    tune_parser.add_argument('--random-state', '-r', type=int, default=42, help='Random seed for reproducibility')
    
    # ML prediction command
    predict_ml_parser = subparsers.add_parser('predict-ml', help='Predict match outcome using ML models')
    predict_ml_parser.add_argument('match_id', help='ID of the match to predict')
    predict_ml_parser.add_argument('--model', '-m', default='ensemble', 
                                  help='ML model to use (default: ensemble of all models)')
    predict_ml_parser.add_argument('--bookmaker-weight', '-b', type=float, default=0.5, 
                                  help='Weight to give bookmaker odds (0-1)')
    
    args = parser.parse_args()
    
    # Initialize the BookieEvaluator
    evaluator = BookieEvaluator()
    
    # Execute the appropriate command
    if args.command == 'fetch':
        matches, odds = evaluator.fetch_upcoming_matches(args.sport, args.competition)
        
        if matches:
            print(f"\n{Fore.CYAN}Upcoming Matches:{Style.RESET_ALL}")
            for match in matches[:10]:  # Show first 10 matches
                home = match.get('homeTeam', {}).get('name', 'Unknown')
                away = match.get('awayTeam', {}).get('name', 'Unknown')
                date = match.get('utcDate', 'Unknown')
                match_id = match.get('id', '')
                
                print(f"{Fore.WHITE}ID: {match_id} - {home} vs {away} - {date}{Style.RESET_ALL}")
            
            if len(matches) > 10:
                print(f"... and {len(matches) - 10} more matches.")
    
    elif args.command == 'analyze':
        # Fetch latest data first
        matches, odds = evaluator.fetch_upcoming_matches()
        
        # Analyze the specified match
        analysis = evaluator.analyze_match(args.match_id, matches, odds)
        
        # Display the analysis
        evaluator.display_analysis(analysis)
        
        # Show historically reliable betting opportunities
        evaluator.identify_reliable_bets(analysis)
    
    elif args.command == 'record':
        evaluator.record_match_result(args.match_id, args.result)
        
        # Show updated outcome reliability stats
        evaluator.display_outcome_reliability_stats()
    
    elif args.command == 'stats':
        evaluator.display_outcome_reliability_stats()
    
    elif args.command == 'reliable':
        # Fetch latest data first
        matches, odds = evaluator.fetch_upcoming_matches()
        
        # Analyze the specified match
        analysis = evaluator.analyze_match(args.match_id, matches, odds)
        
        # Show historically reliable betting opportunities
        evaluator.identify_reliable_bets(analysis, min_accuracy=args.min_accuracy, min_matches=args.min_matches)
    
    elif args.command == 'train-ml':
        evaluator.train_ml_models(test_size=args.test_size, random_state=args.random_state)
    
    elif args.command == 'tune':
        evaluator.hyperparameter_tuning(
            model_type=args.model,
            n_trials=args.trials,
            test_size=args.test_size,
            random_state=args.random_state
        )
    
    elif args.command == 'predict-ml':
        # Fetch latest data first
        matches, odds = evaluator.fetch_upcoming_matches()
        
        # Analyze the specified match
        analysis = evaluator.analyze_match(args.match_id, matches, odds)
        
        # Make ML prediction
        evaluator.predict_with_ml(
            analysis,
            model_name=args.model,
            bookmaker_weight=args.bookmaker_weight
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()