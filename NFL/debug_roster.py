
import nfl_data_py as nfl
import pandas as pd

pd.set_option('display.max_columns', None)

year = 2025
team = 'KC'

print(f"Fetching data for {team} in {year}...")

# 1. Roster
print("Fetching Roster...")
try:
    roster = nfl.import_seasonal_rosters([year])
    print(f"Roster shape: {roster.shape}")
    team_roster = roster[roster['team'] == team]
    print(f"Team Roster shape: {team_roster.shape}")
    if not team_roster.empty:
        print("Roster Columns:", team_roster.columns.tolist())
        print("First 2 rows roster:")
        print(team_roster.head(2)[['player_name', 'player_id', 'position']])
except Exception as e:
    print(f"Roster Error: {e}")

# 2. Stats
print("\nFetching Stats...")
try:
    stats = nfl.import_seasonal_data([year])
    print(f"Stats shape: {stats.shape}")
    team_stats = stats[stats['recent_team'] == team]
    print(f"Team Stats shape: {team_stats.shape}")
    if not team_stats.empty:
        print("Stats Columns:", team_stats.columns.tolist())
        print("First 2 rows stats:")
        print(team_stats.head(2)[['player_name', 'player_id', 'passing_yards']])
except Exception as e:
    print(f"Stats Error: {e}")

# 3. Merge
if 'team_roster' in locals() and 'team_stats' in locals() and not team_roster.empty:
    print("\nAttempting Merge...")
    merged = pd.merge(
        team_roster, 
        team_stats[['player_id', 'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds']], 
        on='player_id', 
        how='left'
    )
    print(f"Merged shape: {merged.shape}")
    print("Merged Head:")
    print(merged[['player_name', 'position', 'passing_yards']].head())
    
    # Check for NaNs in stats
    print("\nStats Null Count in Merged:")
    print(merged[['passing_yards']].isnull().sum())
