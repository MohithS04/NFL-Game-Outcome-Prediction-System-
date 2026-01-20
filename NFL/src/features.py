import pandas as pd

def prepare_features(schedule_df):
    """
    Prepares features for the model from the schedule dataframe.
    """
    # Filter for regular season and completed games or games with lines
    df = schedule_df.copy()
    
    # Basic data cleaning
    # Use ELO ratings provided in the dataset or odds
    
    # Feature selection:
    # home_team, away_team (encoded?)
    # spread_line (market implication)
    # home_rest, away_rest
    # elo1_pre, elo2_pre (if available in schedule data, otherwise we calculate)
    
    # nfl_data_py schedule usually has:
    # game_id, season, game_type, week, gameday, weekday, gametime, away_team, home_team, 
    # away_score, home_score, home_moneyline, away_moneyline, spread_line, etc.
    
    # Target: Home Win (1 if home_score > away_score else 0)
    # Note: Ties are rare but possible.
    
    df['target'] = (df['home_score'] > df['away_score']).astype(int)
    
    # Features for prediction (Pre-game info only)
    features = [
        'spread_line', 
        'home_moneyline', 
        'away_moneyline'
        # Add more advanced features like rolling averages later
    ]
    
    # Drop rows where critical features are missing
    df = df.dropna(subset=features + ['home_score', 'away_score'])
    
    return df[features], df['target'], df
