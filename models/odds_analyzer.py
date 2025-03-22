import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Union
from collections import defaultdict

class OddsAnalyzer:
    """Class for analyzing odds and evaluating bookmakers"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.historical_performance_file = os.path.join(data_dir, 'bookmaker_performance.json')
        self.outcome_performance_file = os.path.join(data_dir, 'outcome_performance.json')
        self.bookmaker_performance = self._load_historical_performance()
        self.outcome_performance = self._load_outcome_performance()
    
    def _load_historical_performance(self) -> Dict:
        """Load historical performance data if available"""
        if os.path.exists(self.historical_performance_file):
            with open(self.historical_performance_file, 'r') as f:
                return json.load(f)
        return {
            'bookmakers': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _load_outcome_performance(self) -> Dict:
        """Load outcome performance data if available"""
        if os.path.exists(self.outcome_performance_file):
            with open(self.outcome_performance_file, 'r') as f:
                return json.load(f)
        return {
            'confidence_bands': {
                'very_high': {'threshold': 0.75, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'high': {'threshold': 0.60, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'medium': {'threshold': 0.45, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'low': {'threshold': 0.30, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'very_low': {'threshold': 0.0, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}}
            },
            'consensus_levels': {
                'strong': {'threshold': 0.10, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'moderate': {'threshold': 0.20, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}},
                'weak': {'threshold': 1.0, 'home': {'correct': 0, 'total': 0}, 'draw': {'correct': 0, 'total': 0}, 'away': {'correct': 0, 'total': 0}}
            },
            'matches': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def save_historical_performance(self):
        """Save the current bookmaker performance data"""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.historical_performance_file, 'w') as f:
            json.dump(self.bookmaker_performance, f, indent=2)
    
    def save_outcome_performance(self):
        """Save the current outcome performance data"""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.outcome_performance_file, 'w') as f:
            json.dump(self.outcome_performance, f, indent=2)
    
    def decimal_to_implied_probability(self, decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        return 1.0 / decimal_odds if decimal_odds > 0 else 0.0
    
    def fractional_to_decimal(self, fractional_odds: str) -> float:
        """Convert fractional odds to decimal format"""
        try:
            if '/' in fractional_odds:
                numerator, denominator = map(float, fractional_odds.split('/'))
                return 1.0 + (numerator / denominator)
            else:
                return float(fractional_odds)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal format"""
        if american_odds > 0:
            return 1.0 + (american_odds / 100.0)
        elif american_odds < 0:
            return 1.0 + (100.0 / abs(american_odds))
        else:
            return 0.0
    
    def calculate_market_margin(self, odds: List[float]) -> float:
        """Calculate the bookmaker's margin in a market"""
        implied_probs = [self.decimal_to_implied_probability(o) for o in odds]
        return sum(implied_probs) - 1.0
    
    def remove_margin(self, odds: List[float]) -> List[float]:
        """Remove the bookmaker's margin to get fair odds"""
        implied_probs = [self.decimal_to_implied_probability(o) for o in odds]
        margin = sum(implied_probs) - 1.0
        
        # Distribute margin proportionally
        fair_probs = [p - (p * margin / sum(implied_probs)) for p in implied_probs]
        fair_odds = [1.0 / p if p > 0 else 0.0 for p in fair_probs]
        
        return fair_odds
    
    def analyze_match_odds(self, match_odds: Dict) -> Dict:
        """Analyze odds for a specific match from different bookmakers"""
        analysis = {
            'match_id': match_odds.get('id', ''),
            'home_team': match_odds.get('home_team', ''),
            'away_team': match_odds.get('away_team', ''),
            'bookmakers': []
        }
        
        for bookmaker in match_odds.get('bookmakers', []):
            bookie_name = bookmaker.get('title', '')
            markets = bookmaker.get('markets', [])
            
            for market in markets:
                if market.get('key') == 'h2h':  # 1X2 market (home, draw, away)
                    outcomes = market.get('outcomes', [])
                    
                    # Extract odds for each outcome
                    home_odds = next((o.get('price', 0) for o in outcomes if o.get('name') == match_odds.get('home_team')), 0)
                    away_odds = next((o.get('price', 0) for o in outcomes if o.get('name') == match_odds.get('away_team')), 0)
                    draw_odds = next((o.get('price', 0) for o in outcomes if o.get('name', '').lower() == 'draw'), 0)
                    
                    if home_odds and away_odds and draw_odds:
                        odds_list = [home_odds, draw_odds, away_odds]
                        margin = self.calculate_market_margin(odds_list)
                        fair_odds = self.remove_margin(odds_list)
                        
                        analysis['bookmakers'].append({
                            'name': bookie_name,
                            'home_odds': home_odds,
                            'draw_odds': draw_odds,
                            'away_odds': away_odds,
                            'margin': margin,
                            'fair_home_odds': fair_odds[0],
                            'fair_draw_odds': fair_odds[1],
                            'fair_away_odds': fair_odds[2],
                            'implied_home_prob': self.decimal_to_implied_probability(home_odds),
                            'implied_draw_prob': self.decimal_to_implied_probability(draw_odds),
                            'implied_away_prob': self.decimal_to_implied_probability(away_odds),
                        })
        
        return analysis
    
    def evaluate_bookmaker_accuracy(self, match_id: str, actual_result: str, match_analysis: Dict) -> None:
        """
        Evaluate how accurate each bookmaker was for a completed match
        
        Parameters:
        - match_id: Unique identifier for the match
        - actual_result: The actual outcome ('home', 'draw', or 'away')
        - match_analysis: The pre-match analysis with bookmaker odds
        """
        if 'bookmakers' not in self.bookmaker_performance:
            self.bookmaker_performance['bookmakers'] = {}
        
        for bookmaker in match_analysis.get('bookmakers', []):
            bookie_name = bookmaker.get('name')
            
            # Initialize bookmaker data if not exists
            if bookie_name not in self.bookmaker_performance['bookmakers']:
                self.bookmaker_performance['bookmakers'][bookie_name] = {
                    'total_matches': 0,
                    'correct_predictions': 0,
                    'log_loss': 0,
                    'brier_score': 0,
                    'margin_avg': 0,
                    'matches': []
                }
            
            # Get implied probabilities
            home_prob = bookmaker.get('implied_home_prob', 0)
            draw_prob = bookmaker.get('implied_draw_prob', 0)
            away_prob = bookmaker.get('implied_away_prob', 0)
            
            # Determine which probability corresponds to the actual result
            if actual_result == 'home':
                predicted_prob = home_prob
                actual_probs = [1, 0, 0]  # One-hot encoded [home, draw, away]
            elif actual_result == 'draw':
                predicted_prob = draw_prob
                actual_probs = [0, 1, 0]
            elif actual_result == 'away':
                predicted_prob = away_prob
                actual_probs = [0, 0, 1]
            else:
                continue  # Skip if the result is invalid
            
            # Calculate prediction metrics
            brier_score = np.mean([(home_prob - actual_probs[0])**2, 
                                  (draw_prob - actual_probs[1])**2, 
                                  (away_prob - actual_probs[2])**2])
            
            # Log loss with clipping to avoid infinity
            eps = 1e-15
            home_prob = np.clip(home_prob, eps, 1-eps)
            draw_prob = np.clip(draw_prob, eps, 1-eps)
            away_prob = np.clip(away_prob, eps, 1-eps)
            log_loss = -(actual_probs[0] * np.log(home_prob) + 
                         actual_probs[1] * np.log(draw_prob) + 
                         actual_probs[2] * np.log(away_prob))
            
            # Update bookmaker performance
            bookmaker_data = self.bookmaker_performance['bookmakers'][bookie_name]
            n = bookmaker_data['total_matches']
            
            # Determine if the bookmaker's highest probability matched the actual result
            predicted_outcome = 'home'
            highest_prob = home_prob
            
            if draw_prob > highest_prob:
                predicted_outcome = 'draw'
                highest_prob = draw_prob
                
            if away_prob > highest_prob:
                predicted_outcome = 'away'
                highest_prob = away_prob
            
            correct_prediction = predicted_outcome == actual_result
            
            # Update running averages
            bookmaker_data['total_matches'] += 1
            bookmaker_data['correct_predictions'] += 1 if correct_prediction else 0
            bookmaker_data['log_loss'] = (n * bookmaker_data['log_loss'] + log_loss) / (n + 1)
            bookmaker_data['brier_score'] = (n * bookmaker_data['brier_score'] + brier_score) / (n + 1)
            bookmaker_data['margin_avg'] = (n * bookmaker_data['margin_avg'] + bookmaker.get('margin', 0)) / (n + 1)
            
            # Add match to history
            bookmaker_data['matches'].append({
                'match_id': match_id,
                'actual_result': actual_result,
                'predicted_outcome': predicted_outcome,
                'correct': correct_prediction,
                'home_prob': home_prob,
                'draw_prob': draw_prob,
                'away_prob': away_prob,
                'log_loss': log_loss,
                'brier_score': brier_score
            })
        
        # Update last updated timestamp
        self.bookmaker_performance['last_updated'] = datetime.now().isoformat()
        self.save_historical_performance()
    
    def get_bookmaker_rankings(self, metric='brier_score', min_matches=10) -> List[Dict]:
        """
        Get bookmakers ranked by accuracy
        
        Parameters:
        - metric: The metric to rank by ('brier_score', 'log_loss', or 'correct_predictions')
        - min_matches: Minimum number of matches required to be included in ranking
        
        Returns:
        - List of bookmakers with their performance metrics, ordered by the specified metric
        """
        rankings = []
        
        for bookie_name, data in self.bookmaker_performance.get('bookmakers', {}).items():
            if data.get('total_matches', 0) >= min_matches:
                accuracy = data.get('correct_predictions', 0) / data.get('total_matches', 1)
                
                rankings.append({
                    'name': bookie_name,
                    'matches': data.get('total_matches', 0),
                    'correct_predictions': data.get('correct_predictions', 0),
                    'accuracy': accuracy,
                    'log_loss': data.get('log_loss', 0),
                    'brier_score': data.get('brier_score', 0),
                    'avg_margin': data.get('margin_avg', 0)
                })
        
        # Sort by the specified metric
        if metric == 'brier_score' or metric == 'log_loss':
            # Lower is better for these metrics
            rankings.sort(key=lambda x: x.get(metric, float('inf')))
        else:
            # Higher is better for accuracy and correct_predictions
            rankings.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        return rankings
    
    def identify_value_bets(self, match_analysis: Dict, confidence_threshold=0.05) -> List[Dict]:
        """
        Identify potential value bets based on bookmaker disagreement
        
        Parameters:
        - match_analysis: Analysis of a match with different bookmaker odds
        - confidence_threshold: Minimum probability difference to consider a value bet
        
        Returns:
        - List of potential value bets
        """
        if not match_analysis.get('bookmakers'):
            return []
        
        # Calculate consensus probabilities (average of all bookmakers)
        all_home_probs = [b.get('implied_home_prob', 0) for b in match_analysis['bookmakers']]
        all_draw_probs = [b.get('implied_draw_prob', 0) for b in match_analysis['bookmakers']]
        all_away_probs = [b.get('implied_away_prob', 0) for b in match_analysis['bookmakers']]
        
        consensus_home_prob = np.mean(all_home_probs) if all_home_probs else 0
        consensus_draw_prob = np.mean(all_draw_probs) if all_draw_probs else 0
        consensus_away_prob = np.mean(all_away_probs) if all_away_probs else 0
        
        # Get top-ranked bookmakers
        top_bookmakers = self.get_bookmaker_rankings(metric='brier_score', min_matches=10)[:5]
        top_bookmaker_names = [b.get('name') for b in top_bookmakers]
        
        # Calculate consensus probabilities from top bookmakers only
        top_bookies_in_match = [b for b in match_analysis['bookmakers'] if b.get('name') in top_bookmaker_names]
        
        if top_bookies_in_match:
            top_home_probs = [b.get('implied_home_prob', 0) for b in top_bookies_in_match]
            top_draw_probs = [b.get('implied_draw_prob', 0) for b in top_bookies_in_match]
            top_away_probs = [b.get('implied_away_prob', 0) for b in top_bookies_in_match]
            
            top_consensus_home_prob = np.mean(top_home_probs)
            top_consensus_draw_prob = np.mean(top_draw_probs)
            top_consensus_away_prob = np.mean(top_away_probs)
        else:
            # Fall back to all bookmakers if no top bookies are available for this match
            top_consensus_home_prob = consensus_home_prob
            top_consensus_draw_prob = consensus_draw_prob
            top_consensus_away_prob = consensus_away_prob
        
        value_bets = []
        
        # Check each bookmaker for value opportunities
        for bookmaker in match_analysis['bookmakers']:
            bookie_name = bookmaker.get('name')
            
            # Skip if this is one of our top trusted bookmakers
            if bookie_name in top_bookmaker_names:
                continue
            
            # Get implied probabilities from this bookmaker
            home_prob = bookmaker.get('implied_home_prob', 0)
            draw_prob = bookmaker.get('implied_draw_prob', 0)
            away_prob = bookmaker.get('implied_away_prob', 0)
            
            # Check for significant disagreements
            home_diff = top_consensus_home_prob - home_prob
            draw_diff = top_consensus_draw_prob - draw_prob
            away_diff = top_consensus_away_prob - away_prob
            
            # If this bookmaker's probability is significantly lower than the consensus
            # of top bookmakers, it might be a value bet opportunity
            if home_diff > confidence_threshold:
                value_bets.append({
                    'type': 'home',
                    'bookmaker': bookie_name,
                    'odds': bookmaker.get('home_odds'),
                    'implied_prob': home_prob,
                    'consensus_prob': top_consensus_home_prob,
                    'diff': home_diff,
                    'expected_value': bookmaker.get('home_odds') * top_consensus_home_prob
                })
                
            if draw_diff > confidence_threshold:
                value_bets.append({
                    'type': 'draw',
                    'bookmaker': bookie_name,
                    'odds': bookmaker.get('draw_odds'),
                    'implied_prob': draw_prob,
                    'consensus_prob': top_consensus_draw_prob,
                    'diff': draw_diff,
                    'expected_value': bookmaker.get('draw_odds') * top_consensus_draw_prob
                })
                
            if away_diff > confidence_threshold:
                value_bets.append({
                    'type': 'away',
                    'bookmaker': bookie_name,
                    'odds': bookmaker.get('away_odds'),
                    'implied_prob': away_prob,
                    'consensus_prob': top_consensus_away_prob,
                    'diff': away_diff,
                    'expected_value': bookmaker.get('away_odds') * top_consensus_away_prob
                })
        
        # Sort by expected value, highest first
        value_bets.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        
        return value_bets
    
    def evaluate_outcome_reliability(self, match_id: str, actual_result: str, match_analysis: Dict) -> None:
        """
        Evaluate the reliability of predicted outcomes for a completed match
        
        Parameters:
        - match_id: Unique identifier for the match
        - actual_result: The actual outcome ('home', 'draw', or 'away')
        - match_analysis: The pre-match analysis with bookmaker odds
        """
        if not match_analysis.get('bookmakers'):
            return
        
        # Calculate average implied probabilities across all bookmakers
        home_probs = [b.get('implied_home_prob', 0) for b in match_analysis['bookmakers']]
        draw_probs = [b.get('implied_draw_prob', 0) for b in match_analysis['bookmakers']]
        away_probs = [b.get('implied_away_prob', 0) for b in match_analysis['bookmakers']]
        
        avg_home_prob = sum(home_probs) / len(home_probs) if home_probs else 0
        avg_draw_prob = sum(draw_probs) / len(draw_probs) if draw_probs else 0
        avg_away_prob = sum(away_probs) / len(away_probs) if away_probs else 0
        
        # Calculate standard deviation to measure consensus
        std_home_prob = np.std(home_probs) if len(home_probs) > 1 else 0
        std_draw_prob = np.std(draw_probs) if len(draw_probs) > 1 else 0
        std_away_prob = np.std(away_probs) if len(away_probs) > 1 else 0
        
        # Determine predicted outcome
        predicted_outcome = 'home'
        highest_prob = avg_home_prob
        
        if avg_draw_prob > highest_prob:
            predicted_outcome = 'draw'
            highest_prob = avg_draw_prob
            
        if avg_away_prob > highest_prob:
            predicted_outcome = 'away'
            highest_prob = avg_away_prob
        
        # Get the standard deviation for the predicted outcome
        outcome_std = std_home_prob
        if predicted_outcome == 'draw':
            outcome_std = std_draw_prob
        elif predicted_outcome == 'away':
            outcome_std = std_away_prob
        
        # Determine consensus level based on standard deviation
        consensus_level = 'weak'
        for level, data in sorted(self.outcome_performance['consensus_levels'].items(), 
                                  key=lambda x: x[1]['threshold']):
            if outcome_std <= data['threshold']:
                consensus_level = level
                break
        
        # Determine confidence band based on probability
        confidence_band = 'very_low'
        for band, data in sorted(self.outcome_performance['confidence_bands'].items(), 
                                 key=lambda x: x[1]['threshold'], reverse=True):
            if highest_prob >= data['threshold']:
                confidence_band = band
                break
        
        # Update outcome performance tracking
        self.outcome_performance['confidence_bands'][confidence_band][predicted_outcome]['total'] += 1
        self.outcome_performance['consensus_levels'][consensus_level][predicted_outcome]['total'] += 1
        
        if predicted_outcome == actual_result:
            self.outcome_performance['confidence_bands'][confidence_band][predicted_outcome]['correct'] += 1
            self.outcome_performance['consensus_levels'][consensus_level][predicted_outcome]['correct'] += 1
        
        # Store match details for future reference
        self.outcome_performance['matches'].append({
            'match_id': match_id,
            'home_team': match_analysis.get('home_team', ''),
            'away_team': match_analysis.get('away_team', ''),
            'match_date': match_analysis.get('match_date', ''),
            'competition': match_analysis.get('competition', ''),
            'avg_home_prob': avg_home_prob,
            'avg_draw_prob': avg_draw_prob,
            'avg_away_prob': avg_away_prob,
            'std_home_prob': std_home_prob,
            'std_draw_prob': std_draw_prob,
            'std_away_prob': std_away_prob,
            'predicted_outcome': predicted_outcome,
            'actual_result': actual_result,
            'correct': predicted_outcome == actual_result,
            'confidence_band': confidence_band,
            'consensus_level': consensus_level
        })
        
        # Update timestamp
        self.outcome_performance['last_updated'] = datetime.now().isoformat()
        self.save_outcome_performance()
    
    def get_outcome_reliability_stats(self) -> Dict:
        """
        Get statistics on outcome prediction reliability
        
        Returns:
        - Dictionary with reliability statistics by outcome type and confidence level
        """
        stats = {
            'by_confidence': {},
            'by_consensus': {},
            'overall': {'home': {'correct': 0, 'total': 0, 'accuracy': 0},
                        'draw': {'correct': 0, 'total': 0, 'accuracy': 0},
                        'away': {'correct': 0, 'total': 0, 'accuracy': 0}}
        }
        
        # Calculate stats by confidence band
        for band, data in self.outcome_performance['confidence_bands'].items():
            stats['by_confidence'][band] = {}
            
            for outcome in ['home', 'draw', 'away']:
                correct = data[outcome]['correct']
                total = data[outcome]['total']
                accuracy = correct / total if total > 0 else 0
                
                stats['by_confidence'][band][outcome] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                }
                
                # Update overall stats
                stats['overall'][outcome]['correct'] += correct
                stats['overall'][outcome]['total'] += total
        
        # Calculate stats by consensus level
        for level, data in self.outcome_performance['consensus_levels'].items():
            stats['by_consensus'][level] = {}
            
            for outcome in ['home', 'draw', 'away']:
                correct = data[outcome]['correct']
                total = data[outcome]['total']
                accuracy = correct / total if total > 0 else 0
                
                stats['by_consensus'][level][outcome] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                }
        
        # Calculate overall accuracy
        for outcome in ['home', 'draw', 'away']:
            stats['overall'][outcome]['accuracy'] = (
                stats['overall'][outcome]['correct'] / stats['overall'][outcome]['total']
                if stats['overall'][outcome]['total'] > 0 else 0
            )
        
        return stats
    
    def identify_reliable_outcomes(self, match_analysis: Dict, min_accuracy=0.7, min_matches=5) -> Dict:
        """
        Identify reliable betting outcomes based on historical pattern matching
        
        Parameters:
        - match_analysis: Analysis of a match with different bookmaker odds
        - min_accuracy: Minimum historical accuracy to consider an outcome reliable
        - min_matches: Minimum number of historical matches to consider a pattern reliable
        
        Returns:
        - Dictionary with recommended outcomes and their reliability metrics
        """
        if not match_analysis.get('bookmakers'):
            return {}
        
        # Calculate average implied probabilities across all bookmakers
        home_probs = [b.get('implied_home_prob', 0) for b in match_analysis['bookmakers']]
        draw_probs = [b.get('implied_draw_prob', 0) for b in match_analysis['bookmakers']]
        away_probs = [b.get('implied_away_prob', 0) for b in match_analysis['bookmakers']]
        
        avg_home_prob = sum(home_probs) / len(home_probs) if home_probs else 0
        avg_draw_prob = sum(draw_probs) / len(draw_probs) if draw_probs else 0
        avg_away_prob = sum(away_probs) / len(away_probs) if away_probs else 0
        
        # Calculate standard deviation to measure consensus
        std_home_prob = np.std(home_probs) if len(home_probs) > 1 else 0
        std_draw_prob = np.std(draw_probs) if len(draw_probs) > 1 else 0
        std_away_prob = np.std(away_probs) if len(away_probs) > 1 else 0
        
        # Determine confidence bands for each outcome
        confidence_bands = {}
        for outcome, prob in [('home', avg_home_prob), ('draw', avg_draw_prob), ('away', avg_away_prob)]:
            for band, data in sorted(self.outcome_performance['confidence_bands'].items(), 
                                     key=lambda x: x[1]['threshold'], reverse=True):
                if prob >= data['threshold']:
                    confidence_bands[outcome] = band
                    break
        
        # Determine consensus levels for each outcome
        consensus_levels = {}
        for outcome, std in [('home', std_home_prob), ('draw', std_draw_prob), ('away', std_away_prob)]:
            for level, data in sorted(self.outcome_performance['consensus_levels'].items(), 
                                      key=lambda x: x[1]['threshold']):
                if std <= data['threshold']:
                    consensus_levels[outcome] = level
                    break
        
        # Get historical stats
        stats = self.get_outcome_reliability_stats()
        
        # Identify reliable outcomes
        reliable_outcomes = {}
        
        for outcome in ['home', 'draw', 'away']:
            if outcome not in confidence_bands or outcome not in consensus_levels:
                continue
                
            conf_band = confidence_bands[outcome]
            cons_level = consensus_levels[outcome]
            
            # Check if we have enough historical data
            conf_stats = stats['by_confidence'].get(conf_band, {}).get(outcome, {})
            cons_stats = stats['by_consensus'].get(cons_level, {}).get(outcome, {})
            
            conf_total = conf_stats.get('total', 0)
            conf_accuracy = conf_stats.get('accuracy', 0)
            
            cons_total = cons_stats.get('total', 0)
            cons_accuracy = cons_stats.get('accuracy', 0)
            
            # Determine if this outcome is historically reliable
            if (conf_total >= min_matches and conf_accuracy >= min_accuracy and
                cons_total >= min_matches and cons_accuracy >= min_accuracy):
                
                # Calculate the probability for this outcome
                prob = avg_home_prob
                if outcome == 'draw':
                    prob = avg_draw_prob
                elif outcome == 'away':
                    prob = avg_away_prob
                
                # Find the average odds for this outcome across bookmakers
                odds_key = 'home_odds'
                if outcome == 'draw':
                    odds_key = 'draw_odds'
                elif outcome == 'away':
                    odds_key = 'away_odds'
                
                odds_values = [b.get(odds_key, 0) for b in match_analysis['bookmakers']]
                avg_odds = sum(odds_values) / len(odds_values) if odds_values else 0
                
                reliable_outcomes[outcome] = {
                    'implied_probability': prob,
                    'average_odds': avg_odds,
                    'confidence_band': conf_band,
                    'consensus_level': cons_level,
                    'historical_accuracy': conf_accuracy,
                    'historical_samples': conf_total,
                    'consensus_accuracy': cons_accuracy,
                    'consensus_samples': cons_total,
                    'expected_value': avg_odds * conf_accuracy
                }
        
        return reliable_outcomes