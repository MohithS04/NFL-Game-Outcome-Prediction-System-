import nfl_data_py as nfl
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(years):
    """
    Loads schedule and weekly data for the specified years.
    """
    print(f"Loading data for years: {years}...")
    
    # Load schedule data
    schedule = nfl.import_schedules(years)
    
    # Load weekly data (player stats aggregated) - might not be strictly needed for team-level prediction
    # but could be useful for advanced features. 
    # For MVP, we primarily need game outcomes and potential team-level aggregated stats if available.
    # nfl_data_py doesn't have a direct 'team_stats' import that is as simple as schedule for outcomes,
    # but we can derive team stats or use other imports if needed.
    # For this MVP, let's stick to schedule which contains scores and ELO ratings often.
    
    return schedule

@st.cache_data
def get_team_desc():
    return nfl.import_team_desc()

@st.cache_data
def get_season_stats(year):
    """
    Aggregates weekly player data to calculate team-level offensive and defensive stats.
    Returns a DataFrame with columns: [off_pass_epa, off_rush_epa, def_pass_epa, def_rush_epa]
    """
    print(f"Loading weekly data for stats analysis: {year}...")
    try:
        weekly = nfl.import_weekly_data([year])
        
        # Offense: Group by recent_team
        # EPA is total per player, we sum per team-week then average per season? 
        # Or just sum all and divide by games? easier to sum all then normalize.
        # Note: weekly data is player level. Summing player EPAs gives approx team EPA.
        
        offense = weekly.groupby('recent_team')[['passing_epa', 'rushing_epa']].sum()
        # count games? approximation: unique weeks per team
        games_played = weekly.groupby('recent_team')['week'].nunique()
        offense = offense.div(games_played, axis=0)
        offense.columns = ['off_pass_epa', 'off_rush_epa']
        
        # Defense: Group by opponent_team
        defense = weekly.groupby('opponent_team')[['passing_epa', 'rushing_epa']].sum()
        games_played_def = weekly.groupby('opponent_team')['week'].nunique()
        defense = defense.div(games_played_def, axis=0)
        defense.columns = ['def_pass_epa', 'def_rush_epa']
        
        # Combine
        stats = pd.concat([offense, defense], axis=1)
        
        # Ranks (Lower Rank #1 is better for Offense, Lower Allowed EPA is better for Defense? 
        # Actually EPA: Higher is better for Offense. Lower (negative) is better for Defense (usually).
        # But 'Allowed EPA' being high means bad defense.
        # So for Ranking:
        # Offense: Ascending=False (High is #1)
        # Defense: Ascending=True (Low is #1) - Wait, EPA allowed. High EPA allowed = Bad Defense.
        # So Rank #1 Defense should allow Lowest EPA.
        
        stats['off_pass_rank'] = stats['off_pass_epa'].rank(ascending=False)
        stats['off_rush_rank'] = stats['off_rush_epa'].rank(ascending=False)
        stats['def_pass_rank'] = stats['def_pass_epa'].rank(ascending=True) # Lowest allowed is best
        stats['def_rush_rank'] = stats['def_rush_epa'].rank(ascending=True)
        
        return stats
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return pd.DataFrame()

@st.cache_data
def get_recent_form(year, team, week=None):
    """
    Calculates the recent form (Last 5 games win %) for a team.
    If week is specified, only looks at games before that week.
    """
    try:
        schedule = nfl.import_schedules([year])
        
        # Filter for team
        team_games = schedule[(schedule['home_team'] == team) | (schedule['away_team'] == team)]
        
        # Filter completed games
        played = team_games.dropna(subset=['home_score', 'away_score'])
        
        if week:
            played = played[played['week'] < week]
            
        # Sort by week ascending
        played = played.sort_values('week')
        
        # Take last 5
        last_5 = played.tail(5)
        
        if last_5.empty:
            return 0.0, []
            
        wins = 0
        results = []
        for index, row in last_5.iterrows():
            if row['home_team'] == team:
                won = row['home_score'] > row['away_score']
            else:
                won = row['away_score'] > row['home_score']
            
            if won: wins += 1
            results.append("W" if won else "L")
            
        return wins / len(last_5), results
    except Exception:
        return 0.0, []

@st.cache_data
def get_top_players(year, team):
    """
    Fetches the top players (Passing, Rushing, Receiving) for a specific team in the given year.
    Returns a dictionary with best players in each category.
    """
    try:
        data = nfl.import_seasonal_data([year])
        team_data = data[data['recent_team'] == team]
        
        if team_data.empty:
            return {}
            
        leaders = {}
        
        # Helper to extract info
        def extract_info(player_row):
            photo = player_row.get('headshot_url', None)
            return {
                'name': player_row['player_name'],
                'photo': photo,
                'stats': player_row
            }

        # Passing Leader
        passer = team_data.sort_values('passing_yards', ascending=False).iloc[0]
        if passer['passing_yards'] > 0:
            leaders['Passing'] = extract_info(passer)
            leaders['Passing']['details'] = f"{int(passer['passing_yards'])} yds, {int(passer['passing_tds'])} TDs"
            
        # Rushing Leader
        runner = team_data.sort_values('rushing_yards', ascending=False).iloc[0]
        if runner['rushing_yards'] > 0:
            leaders['Rushing'] = extract_info(runner)
            leaders['Rushing']['details'] = f"{int(runner['rushing_yards'])} yds, {int(runner['rushing_tds'])} TDs"
            
        # Receiving Leader
        receiver = team_data.sort_values('receiving_yards', ascending=False).iloc[0]
        if receiver['receiving_yards'] > 0:
            leaders['Receiving'] = extract_info(receiver)
            leaders['Receiving']['details'] = f"{int(receiver['receiving_yards'])} yds, {int(receiver['receiving_tds'])} TDs"
            
        return leaders
    except Exception as e:
        print(f"Error fetching top players: {e}")
        return {}

@st.cache_data
def get_team_roster_stats(year, team):
    """
    Fetches the full roster stats for a team in a given year.
    Returns a DataFrame with player details, headshot_url, and key stats.
    """
    try:
        data = nfl.import_seasonal_data([year])
        team_data = data[data['recent_team'] == team].copy()
        
        if team_data.empty:
            return pd.DataFrame()
            
        # Select relevant columns for display
        cols = [
            'headshot_url', 'player_name', 'position', 'games',
            'passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds',
            'receiving_yards', 'receiving_tds', 'receptions'
        ]
        
        # Filter cols that exist
        cols = [c for c in cols if c in team_data.columns]
        
        return team_data[cols]
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return pd.DataFrame()

@st.cache_data
def get_full_roster(year, team):
    """
    Fetches the complete roster for a team in a given year, merging bio and stats.
    Returns DataFrame with detailed player info and key performance metrics.
    """
    try:
        # 1. Fetch Roster (Bio)
        rosters = nfl.import_seasonal_rosters([year])
        team_roster = rosters[rosters['team'] == team].copy()
        
        if team_roster.empty:
            return pd.DataFrame()

        # 2. Fetch Stats (Handle cases where stats are missing, e.g. future/current seasons)
        try:
            stats = nfl.import_seasonal_data([year])
            stats = stats[stats['recent_team'] == team]
        except Exception:
            # If stats fetch fails (e.g. 404 for 2025), use empty DF
            stats = pd.DataFrame()

        # 3. Merge
        if not stats.empty:
            merged = pd.merge(
                team_roster, 
                stats[['player_id', 'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds']], 
                on='player_id', 
                how='left'
            )
        else:
            merged = team_roster

        # Fill NaN stats with 0 (or create cols if they don't exist)
        stat_cols = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds']
        for c in stat_cols:
            if c not in merged.columns:
                merged[c] = 0
            else:
                merged[c] = merged[c].fillna(0)
            
        cols = [
            'headshot_url', 'player_name', 'position', 'jersey_number', 
            'age', 'height', 'weight', 'college', 'years_exp',
            'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds'
        ]
        
        # Ensure cols exist
        cols = [c for c in cols if c in merged.columns]
        
        return merged[cols].sort_values(['position', 'player_name'])
        
    except Exception as e:
        print(f"Error fetching full roster: {e}")
        return pd.DataFrame()
