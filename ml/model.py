import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm

class FootballMatchPredictor:
    """Machine learning model for predicting football match outcomes"""
    
    def __init__(self, data_dir='data', model_dir='ml/models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_dir), exist_ok=True)
    
    def extract_features_from_match(self, match_data: Dict) -> pd.DataFrame:
        """
        Extract features from a single match for prediction
        
        Parameters:
        - match_data: Dictionary containing match analysis data
        
        Returns:
        - DataFrame with features for the match
        """
        features = {}
        
        bookmakers = match_data.get('bookmakers', [])
        if not bookmakers:
            return pd.DataFrame()
        
        # Calculate consensus probabilities
        home_probs = [b.get('implied_home_prob', 0) for b in bookmakers]
        draw_probs = [b.get('implied_draw_prob', 0) for b in bookmakers]
        away_probs = [b.get('implied_away_prob', 0) for b in bookmakers]
        
        features['avg_home_prob'] = np.mean(home_probs) if home_probs else 0
        features['avg_draw_prob'] = np.mean(draw_probs) if draw_probs else 0
        features['avg_away_prob'] = np.mean(away_probs) if away_probs else 0
        
        features['std_home_prob'] = np.std(home_probs) if len(home_probs) > 1 else 0
        features['std_draw_prob'] = np.std(draw_probs) if len(draw_probs) > 1 else 0
        features['std_away_prob'] = np.std(away_probs) if len(away_probs) > 1 else 0
        
        features['max_home_prob'] = max(home_probs) if home_probs else 0
        features['max_draw_prob'] = max(draw_probs) if draw_probs else 0
        features['max_away_prob'] = max(away_probs) if away_probs else 0
        
        features['min_home_prob'] = min(home_probs) if home_probs else 0
        features['min_draw_prob'] = min(draw_probs) if draw_probs else 0
        features['min_away_prob'] = min(away_probs) if away_probs else 0
        
        # Calculate average margins and odds
        margins = [b.get('margin', 0) for b in bookmakers]
        features['avg_margin'] = np.mean(margins) if margins else 0
        features['std_margin'] = np.std(margins) if len(margins) > 1 else 0
        
        # Probability ratios
        if features['avg_away_prob'] > 0:
            features['home_away_ratio'] = features['avg_home_prob'] / features['avg_away_prob']
        else:
            features['home_away_ratio'] = 10.0  # arbitrary high value
            
        if features['avg_draw_prob'] > 0:
            features['home_draw_ratio'] = features['avg_home_prob'] / features['avg_draw_prob']
            features['away_draw_ratio'] = features['avg_away_prob'] / features['avg_draw_prob']
        else:
            features['home_draw_ratio'] = 10.0
            features['away_draw_ratio'] = 10.0
            
        # Favorite indicator (highest probability)
        probs = [features['avg_home_prob'], features['avg_draw_prob'], features['avg_away_prob']]
        max_prob_index = probs.index(max(probs))
        features['favorite_home'] = 1 if max_prob_index == 0 else 0
        features['favorite_draw'] = 1 if max_prob_index == 1 else 0
        features['favorite_away'] = 1 if max_prob_index == 2 else 0
        
        # Calculate odds disagreement metrics
        if len(bookmakers) > 1:
            features['home_disagreement'] = max(home_probs) - min(home_probs)
            features['draw_disagreement'] = max(draw_probs) - min(draw_probs)
            features['away_disagreement'] = max(away_probs) - min(away_probs)
        else:
            features['home_disagreement'] = 0
            features['draw_disagreement'] = 0
            features['away_disagreement'] = 0
        
        # Match metadata (can be expanded)
        features['match_id'] = match_data.get('match_id', '')
        features['home_team'] = match_data.get('home_team', '')
        features['away_team'] = match_data.get('away_team', '')
        features['competition'] = match_data.get('competition', '')
        
        # Convert to DataFrame
        return pd.DataFrame([features])
    
    def prepare_training_data(self, matches: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features from historical matches for model training
        
        Parameters:
        - matches: List of dictionaries containing match data with results
        
        Returns:
        - Tuple of (X, y) for training
        """
        if not matches:
            return pd.DataFrame(), pd.Series()
        
        features_list = []
        outcomes = []
        
        for match in matches:
            actual_result = match.get('actual_result')
            if not actual_result or actual_result not in ['home', 'draw', 'away']:
                continue  # Skip matches without valid results
                
            features = self.extract_features_from_match(match)
            
            if not features.empty:
                features_list.append(features)
                outcomes.append(actual_result)
        
        if not features_list:
            return pd.DataFrame(), pd.Series()
        
        X = pd.concat(features_list, ignore_index=True)
        y = pd.Series(outcomes)
        
        # Keep metadata columns separate
        metadata_cols = ['match_id', 'home_team', 'away_team', 'competition']
        metadata = X[metadata_cols].copy()
        X = X.drop(columns=metadata_cols)
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42) -> Dict:
        """
        Train multiple machine learning models for outcome prediction
        
        Parameters:
        - X: Feature DataFrame
        - y: Target Series (match outcomes)
        - test_size: Proportion of data to use for testing
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with model evaluation metrics
        """
        if X.empty or y.empty:
            return {"error": "No training data available"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Save the scaler
        self.scalers['standard'] = scaler
        joblib.dump(scaler, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        self.model_dir, 'standard_scaler.joblib'))
        
        # Train different models
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=random_state),
            'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Save the model
            self.models[name] = model
            joblib.dump(model, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          self.model_dir, f'{name}_model.joblib'))
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate Brier score for multiclass
            brier = self._compute_multiclass_brier_score(y_test, y_prob, model.classes_)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'brier_score': brier
            }
            
            # Extract feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = feature_importance
            elif name == 'logistic_regression':
                # For logistic regression, we can get coefficients
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': np.abs(model.coef_[0])  # Use first class for simple view
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = feature_importance
        
        # Perform cross-validation for robustness checking
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=5)
            results[name]['cross_val_scores'] = cv_scores.tolist()
            results[name]['cross_val_mean'] = cv_scores.mean()
            results[name]['cross_val_std'] = cv_scores.std()
        
        return results
    
    def hypertune_model(self, X: pd.DataFrame, y: pd.Series, model_type='xgboost', 
                        n_trials=100, test_size=0.2, random_state=42) -> Dict:
        """
        Perform hyperparameter tuning using Optuna
        
        Parameters:
        - X: Feature DataFrame
        - y: Target Series
        - model_type: Type of model to optimize ('xgboost', 'lightgbm', etc.)
        - n_trials: Number of optimization trials
        - test_size: Proportion of data to use for testing
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with tuning results and best parameters
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Define optimization function
        def objective(trial):
            params = {}
            
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': random_state
                }
                model = xgb.XGBClassifier(**params)
            
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
                    'random_state': random_state
                }
                model = lgb.LGBMClassifier(**params)
            
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'random_state': random_state
                }
                model = RandomForestClassifier(**params)
                
            else:
                # Default to simple model
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'max_iter': 1000,
                    'random_state': random_state
                }
                model = LogisticRegression(**params)
            
            # Evaluate model using cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            return cv_scores.mean()
        
        # Create and run optimization study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Train model with best parameters
        if model_type == 'xgboost':
            best_model = xgb.XGBClassifier(**best_params)
        elif model_type == 'lightgbm':
            best_model = lgb.LGBMClassifier(**best_params)
        elif model_type == 'random_forest':
            best_model = RandomForestClassifier(**best_params)
        else:
            best_model = LogisticRegression(**best_params)
        
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        brier = self._compute_multiclass_brier_score(y_test, y_prob, best_model.classes_)
        
        # Save the optimized model
        model_name = f'optimized_{model_type}'
        self.models[model_name] = best_model
        joblib.dump(best_model, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           self.model_dir, f'{model_name}_model.joblib'))
        
        # Extract feature importance if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance[model_name] = feature_importance
        
        return {
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'brier_score': brier,
            'optimization_history': study.trials_dataframe().to_dict(orient='records')
        }
    
    def predict_match(self, match_data: Dict, model_name='ensemble') -> Dict:
        """
        Predict the outcome of a match using trained models
        
        Parameters:
        - match_data: Dictionary containing match analysis data
        - model_name: Name of the model to use for prediction
        
        Returns:
        - Dictionary with predicted probabilities and confidence
        """
        features = self.extract_features_from_match(match_data)
        
        if features.empty:
            return {"error": "Failed to extract features from match data"}
        
        # Keep metadata separately
        metadata_cols = ['match_id', 'home_team', 'away_team', 'competition']
        metadata = {col: features[col].iloc[0] for col in metadata_cols if col in features.columns}
        features = features.drop(columns=metadata_cols, errors='ignore')
        
        # Scale features
        scaler = self.scalers.get('standard')
        if scaler:
            features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
        else:
            # Try to load scaler
            try:
                scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          self.model_dir, 'standard_scaler.joblib')
                scaler = joblib.load(scaler_path)
                features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
                self.scalers['standard'] = scaler
            except:
                # If no scaler is available, use unscaled features
                features_scaled = features
        
        if model_name == 'ensemble':
            # Use an ensemble of all available models
            all_probs = []
            for name, model in self.models.items():
                if model is None:
                    # Try to load the model
                    try:
                        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                self.model_dir, f'{name}_model.joblib')
                        model = joblib.load(model_path)
                        self.models[name] = model
                    except:
                        continue
                
                if model is not None:
                    try:
                        probs = model.predict_proba(features_scaled)
                        all_probs.append(probs)
                    except:
                        pass
            
            if not all_probs:
                return {"error": "No models available for prediction"}
            
            # Average predictions across models
            avg_probs = np.mean(all_probs, axis=0)
            classes = np.array(['home', 'draw', 'away'])  # Assumed order
            
            # Find highest probability outcome
            highest_prob_idx = np.argmax(avg_probs[0])
            predicted_outcome = classes[highest_prob_idx]
            
            # Create prediction result
            prediction = {
                'home_prob': float(avg_probs[0][0]),
                'draw_prob': float(avg_probs[0][1]),
                'away_prob': float(avg_probs[0][2]),
                'predicted_outcome': predicted_outcome,
                'confidence': float(avg_probs[0][highest_prob_idx]),
                'model': 'ensemble'
            }
        else:
            # Use a specific model
            model = self.models.get(model_name)
            
            if model is None:
                # Try to load the model
                try:
                    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            self.model_dir, f'{model_name}_model.joblib')
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                except:
                    return {"error": f"Model '{model_name}' not found"}
            
            # Make prediction
            probs = model.predict_proba(features_scaled)
            predicted_class_idx = np.argmax(probs[0])
            predicted_outcome = model.classes_[predicted_class_idx]
            
            # Map probabilities to outcomes
            outcome_probs = {}
            for i, outcome in enumerate(model.classes_):
                outcome_probs[f'{outcome}_prob'] = float(probs[0][i])
            
            # Create prediction result
            prediction = {
                **outcome_probs,
                'predicted_outcome': predicted_outcome,
                'confidence': float(probs[0][predicted_class_idx]),
                'model': model_name
            }
        
        # Add metadata back to result
        prediction.update(metadata)
        
        return prediction
    
    def combine_predictions(self, match_data: Dict, bookmaker_weight=0.5) -> Dict:
        """
        Combine ML model predictions with bookmaker odds for improved accuracy
        
        Parameters:
        - match_data: Dictionary containing match analysis data
        - bookmaker_weight: Weight to give to bookmaker odds (0-1)
        
        Returns:
        - Dictionary with combined prediction probabilities
        """
        # Get ML prediction
        ml_prediction = self.predict_match(match_data)
        
        if 'error' in ml_prediction:
            return ml_prediction
        
        # Get bookmaker probabilities
        bookmakers = match_data.get('bookmakers', [])
        if not bookmakers:
            return ml_prediction  # No bookmaker data, return ML prediction only
        
        # Calculate average bookmaker probabilities
        home_probs = [b.get('implied_home_prob', 0) for b in bookmakers]
        draw_probs = [b.get('implied_draw_prob', 0) for b in bookmakers]
        away_probs = [b.get('implied_away_prob', 0) for b in bookmakers]
        
        avg_home_prob = np.mean(home_probs) if home_probs else 0
        avg_draw_prob = np.mean(draw_probs) if draw_probs else 0
        avg_away_prob = np.mean(away_probs) if away_probs else 0
        
        # Combine predictions with weighted average
        ml_weight = 1 - bookmaker_weight
        
        combined_home = (ml_prediction.get('home_prob', 0) * ml_weight) + (avg_home_prob * bookmaker_weight)
        combined_draw = (ml_prediction.get('draw_prob', 0) * ml_weight) + (avg_draw_prob * bookmaker_weight)
        combined_away = (ml_prediction.get('away_prob', 0) * ml_weight) + (avg_away_prob * bookmaker_weight)
        
        # Normalize probabilities to sum to 1
        total = combined_home + combined_draw + combined_away
        if total > 0:
            combined_home /= total
            combined_draw /= total
            combined_away /= total
        
        # Determine predicted outcome
        outcomes = ['home', 'draw', 'away']
        combined_probs = [combined_home, combined_draw, combined_away]
        predicted_outcome = outcomes[np.argmax(combined_probs)]
        confidence = max(combined_probs)
        
        # Create combined prediction result
        combined_prediction = {
            'home_prob': combined_home,
            'draw_prob': combined_draw,
            'away_prob': combined_away,
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'ml_prediction': {
                'home_prob': ml_prediction.get('home_prob', 0),
                'draw_prob': ml_prediction.get('draw_prob', 0),
                'away_prob': ml_prediction.get('away_prob', 0),
                'predicted_outcome': ml_prediction.get('predicted_outcome', ''),
                'model': ml_prediction.get('model', '')
            },
            'bookmaker_prediction': {
                'home_prob': avg_home_prob,
                'draw_prob': avg_draw_prob,
                'away_prob': avg_away_prob
            },
            'model': 'combined',
            'bookmaker_weight': bookmaker_weight
        }
        
        # Add metadata if available
        for key in ['match_id', 'home_team', 'away_team', 'competition']:
            if key in ml_prediction:
                combined_prediction[key] = ml_prediction[key]
        
        return combined_prediction
    
    def plot_feature_importance(self, model_name='ensemble', top_n=10):
        """
        Plot feature importance for a specified model
        
        Parameters:
        - model_name: Name of the model to plot feature importance for
        - top_n: Number of top features to display
        """
        if model_name == 'ensemble':
            # Aggregate feature importance across all models
            all_importances = pd.DataFrame()
            
            for name, importance_df in self.feature_importance.items():
                if all_importances.empty:
                    all_importances = importance_df.copy()
                    all_importances.rename(columns={'importance': name}, inplace=True)
                else:
                    all_importances = all_importances.merge(
                        importance_df, on='feature', how='outer'
                    )
                    all_importances.rename(columns={'importance': name}, inplace=True)
            
            if all_importances.empty:
                print("No feature importance data available")
                return
                
            # Calculate average importance
            importance_cols = [col for col in all_importances.columns if col != 'feature']
            all_importances['avg_importance'] = all_importances[importance_cols].mean(axis=1)
            all_importances = all_importances.sort_values('avg_importance', ascending=False)
            
            # Plot top N features
            top_features = all_importances.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['avg_importance'])
            plt.xlabel('Average Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Features (Averaged Across Models)')
            plt.tight_layout()
            plt.show()
            
        else:
            # Plot importance for a specific model
            importance_df = self.feature_importance.get(model_name)
            
            if importance_df is None or importance_df.empty:
                print(f"No feature importance data available for model '{model_name}'")
                return
            
            # Plot top N features
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Features for {model_name}')
            plt.tight_layout()
            plt.show()
    
    def save_shap_analysis(self, X: pd.DataFrame, model_name='xgboost', max_display=20):
        """
        Generate and save SHAP analysis for model interpretability
        
        Parameters:
        - X: Feature DataFrame to use for SHAP analysis
        - model_name: Name of the model to analyze
        - max_display: Maximum number of features to display in plots
        """
        model = self.models.get(model_name)
        
        if model is None:
            # Try to load the model
            try:
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        self.model_dir, f'{model_name}_model.joblib')
                model = joblib.load(model_path)
                self.models[model_name] = model
            except:
                print(f"Model '{model_name}' not found")
                return
        
        # Ensure X is scaled appropriately
        scaler = self.scalers.get('standard')
        if scaler:
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        else:
            X_scaled = X
        
        # Create explainer
        if model_name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_scaled)
        
        # Create output directory
        shap_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              self.model_dir, 'shap_analysis')
        os.makedirs(shap_dir, exist_ok=True)
        
        # Save summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_scaled, plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'{model_name}_shap_summary.png'))
        plt.close()
        
        # Save detailed summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_scaled, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, f'{model_name}_shap_detailed.png'))
        plt.close()
        
        # Save dependency plots for top features
        if self.feature_importance.get(model_name) is not None:
            top_features = self.feature_importance[model_name]['feature'].head(5).tolist()
            
            for feature in top_features:
                if feature in X.columns:
                    plt.figure(figsize=(10, 7))
                    shap.dependence_plot(feature, shap_values, X_scaled, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(shap_dir, f'{model_name}_shap_{feature}.png'))
                    plt.close()
        
        print(f"SHAP analysis saved to {shap_dir}")
    
    def _compute_multiclass_brier_score(self, y_true, y_prob, classes):
        """Compute Brier score for multiclass problems"""
        # Convert y_true to one-hot encoding
        y_true_encoded = np.zeros((len(y_true), len(classes)))
        for i, c in enumerate(y_true):
            class_idx = np.where(classes == c)[0][0]
            y_true_encoded[i, class_idx] = 1
        
        # Calculate Brier score
        return np.mean(np.sum((y_prob - y_true_encoded) ** 2, axis=1)) / len(classes)