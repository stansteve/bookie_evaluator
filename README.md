# Bookie Evaluator

A sophisticated tool for analyzing football match outcomes by evaluating bookmaker odds and using machine learning to identify reliable betting opportunities.

## Overview

Bookie Evaluator analyzes bookmakers' odds for football matches to identify scenarios where bookmakers consistently make accurate predictions. It enhances this analysis with machine learning models trained on historical data to provide more accurate forecasts.

The tool helps you:
- Identify outcomes where bookmakers have historically been accurate
- Find potential value bets based on consensus analysis
- Train and use machine learning models to improve prediction accuracy
- Combine machine learning predictions with bookmaker odds for optimal results

## Features

- **Match Analysis**: Analyze odds from multiple bookmakers for upcoming matches
- **Outcome Reliability**: Track which types of outcomes bookmakers predict accurately
- **Pattern Recognition**: Identify statistical patterns where predictions are most reliable
- **Machine Learning**: Train models on historical data to enhance prediction accuracy
- **Value Betting**: Find potential value bets where probabilities are mispriced
- **Hyperparameter Tuning**: Optimize machine learning models for maximum accuracy
- **Prediction Explanations**: Understand which factors influence predictions using SHAP analysis

## Installation

### Prerequisites

- Python 3.10 or newer
- Poetry (for dependency management)

### Setting up with Poetry

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bookie-evaluator.git
cd bookie-evaluator
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up your API keys in a `.env` file:
```
ODDS_API_KEY=your_odds_api_key
FOOTBALL_DATA_API_KEY=your_football_data_api_key
```

You'll need to register for free API keys from:
- [The Odds API](https://the-odds-api.com/)
- [Football-Data.org](https://www.football-data.org/)

## Usage

Bookie Evaluator provides a command-line interface with several commands:

### Fetch upcoming matches

```bash
poetry run python main.py fetch --sport soccer --competition PL
```

### Analyze a specific match

```bash
poetry run python main.py analyze MATCH_ID
```

### Record match results (to build historical database)

```bash
poetry run python main.py record MATCH_ID home|draw|away
```

### View outcome reliability statistics

```bash
poetry run python main.py stats
```

### Find reliable betting opportunities

```bash
poetry run python main.py reliable MATCH_ID
```

### Train machine learning models

```bash
poetry run python main.py train-ml
```

### Optimize model with hyperparameter tuning

```bash
poetry run python main.py tune --model xgboost --trials 100
```

### Make ML-enhanced predictions

```bash
poetry run python main.py predict-ml MATCH_ID --bookmaker-weight 0.6
```

## Machine Learning Approach

The machine learning system:

1. Extracts features from bookmaker odds, including:
   - Consensus probabilities
   - Bookmaker disagreement metrics
   - Probability ratios
   - Favorite indicators

2. Trains multiple models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM

3. Combines models in an ensemble for robust predictions

4. Fuses ML predictions with bookmaker odds using adjustable weights

## Typical Workflow

1. **Data Collection**:
   - Fetch upcoming matches
   - Analyze matches of interest

2. **Record Results**:
   - After matches complete, record results to build your database

3. **Model Training**:
   - Train ML models on your growing historical database
   - Optimize models with hyperparameter tuning

4. **Make Predictions**:
   - Use combined bookmaker analysis and ML to identify reliable bets
   - Adjust the bookmaker weight based on your confidence in the ML models

## Project Structure

```
bookie_evaluator/
├── main.py                 # Main application and CLI
├── api/
│   └── odds_client.py      # API clients for odds data
├── data/                   # Data storage directory
├── ml/
│   ├── __init__.py         # ML module initialization
│   └── model.py            # Machine learning models
├── models/
│   └── odds_analyzer.py    # Odds analysis logic
└── utils/
    └── file_utils.py       # File handling utilities
```

## Dependencies

This project uses the following key libraries:
- pandas, numpy: Data processing
- scikit-learn: Machine learning core
- xgboost, lightgbm: Advanced ML models
- optuna: Hyperparameter optimization
- shap: Model explainability
- matplotlib: Visualization
- tabulate, colorama: CLI display

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.