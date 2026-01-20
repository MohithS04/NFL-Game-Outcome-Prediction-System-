import streamlit as st
import pandas as pd
import numpy as np
from src.data import load_data, get_team_desc, get_season_stats, get_recent_form, get_top_players, get_team_roster_stats, get_full_roster
from src.features import prepare_features
from src.model import load_trained_model, predict_game
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NFL Predictor", layout="wide")

st.title("NFL Game Outcome Prediction")

@st.cache_resource
def get_model():
    return load_trained_model()

# Custom CSS for animated background and glassmorphism
page_bg_css = """
<style>
/* Animated Football Field Background */
[data-testid="stAppViewContainer"] {
    background-color: #1a422a; /* Deep Grass Green */
    background-image: 
        linear-gradient(rgba(0, 20, 0, 0.85), rgba(0, 20, 0, 0.85)), /* Dark overlay for contrast */
        repeating-linear-gradient(0deg, transparent, transparent 195px, rgba(255, 255, 255, 0.2) 198px, rgba(255, 255, 255, 0.2) 200px);
    background-size: 100% 100%, 100% 1000px;
    animation: fieldScroll 30s linear infinite;
    background-attachment: fixed;
}

@keyframes fieldScroll {
    0% { background-position: 0% 0%; }
    100% { background-position: 0% 1000px; }
}

/* Glassmorphism for main containers */
div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    background: rgba(40, 40, 40, 0.6);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Header transparency */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Metric styling */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(50, 255, 50, 0.8);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

model = get_model()
team_desc = get_team_desc()

# Sidebar for controls
st.sidebar.header("Settings")
available_years = list(range(2025, 2013, -1))
current_year = st.sidebar.selectbox("Select Season", available_years, index=0)

# Load data for the selected season
# Load data for the selected season
try:
    schedule = load_data([current_year])
    season_stats = get_season_stats(current_year)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Helper to get logo
def get_logo(team_abbr, team_df):
    row = team_df[team_df['team_abbr'] == team_abbr]
    if not row.empty:
        return row.iloc[0]['team_logo_espn']
    return None

def moneyline_to_prob(ml):
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return (-ml) / (-ml + 100)

def display_game_card(row, show_result=False, stats=None):
    home_logo = get_logo(row['home_team'], team_desc)
    away_logo = get_logo(row['away_team'], team_desc)
    
    with st.container():
        # Top Row: Logos and Basic Info
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if away_logo:
                st.image(away_logo, width=50)
            st.write(f"**{row['away_team']}**")
            if show_result:
                st.write(f"{int(row['away_score'])}")
        
        with col2:
            st.write("vs")
            if not show_result:
                st.caption(f"Spread: {row['spread_line']}")
        
        with col3:
            if home_logo:
                st.image(home_logo, width=50)
            st.write(f"**{row['home_team']}**")
            if show_result:
                st.write(f"{int(row['home_score'])}")
        
        with col4:
            st.metric("Model Prob", f"{row['Home Win Prob']:.1%}")
            if not show_result and 'Implied Prob' in row:
                st.caption(f"Implied: {row['Implied Prob']:.1%}")
        
        with col5:
            if show_result:
                correct = row['Prediction Correct']
                if correct:
                    st.success("Correct")
                else:
                    st.error("Incorrect")
            else:
                if 'Edge' in row:
                    edge = row['Edge']
                    st.metric("Edge", f"{edge:.1%}", delta=f"{edge:.1%}", delta_color="normal" if edge > 0 else "off")
                    if row.get('Value Bet'):
                        st.success("Value Bet!")
        
        # Win Probability Ratio Bar
        home_prob = row['Home Win Prob']
        away_prob = 1 - home_prob
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="flex: 1; text-align: right; font-size: 0.8em; margin-right: 5px;">{away_prob:.1%}</div>
                <div style="flex: 8; height: 10px; background: #ddd; border-radius: 5px; overflow: hidden; display: flex;">
                    <div style="width: {away_prob*100}%; background: #e73c7e; height: 100%;"></div>
                    <div style="width: {home_prob*100}%; background: #23a6d5; height: 100%;"></div>
                </div>
                <div style="flex: 1; text-align: left; font-size: 0.8em; margin-left: 5px;">{home_prob:.1%}</div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.7em; color: #aaa;">
                <span>{row['away_team']} Win Prob</span>
                <span>{row['home_team']} Win Prob</span>
            </div>
        """, unsafe_allow_html=True)

        
        # Detailed Report Section
        if stats is not None:
            h_sym = row['home_team']
            a_sym = row['away_team']
            
            if h_sym in stats.index and a_sym in stats.index:
                hs = stats.loc[h_sym]
                as_ = stats.loc[a_sym]
                
                with st.expander("Detailed Report (Fitness & Strategy)"):
                    # 1. Recent Form (Fitness)
                    curr_week = row['week'] if 'week' in row else None
                    h_form_score, h_form_rec = get_recent_form(row['season'], h_sym, curr_week)
                    a_form_score, a_form_rec = get_recent_form(row['season'], a_sym, curr_week)
                    
                    st.subheader("Team Fitness (Recent Form - Last 5 Games)")
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        st.write(f"**{a_sym}**: {' '.join(a_form_rec)} ({a_form_score:.0%})")
                        st.progress(a_form_score)
                    with fcol2:
                        st.write(f"**{h_sym}**: {' '.join(h_form_rec)} ({h_form_score:.0%})")
                        st.progress(h_form_score)
                    
                    st.divider()

                    # 2. Detailed Matchup Stats
                    st.subheader("Matchup Strategy")
                    scol1, scol2 = st.columns(2)
                    
                    with scol1:
                        st.markdown(f"**{a_sym} Offense vs {h_sym} Defense**")
                        st.markdown(f"- Pass: Rank #{int(as_['off_pass_rank'])} vs #{int(hs['def_pass_rank'])}")
                        st.markdown(f"- Rush: Rank #{int(as_['off_rush_rank'])} vs #{int(hs['def_rush_rank'])}")
                        
                        if as_['off_pass_rank'] < 10 and hs['def_pass_rank'] > 20:
                            st.info(f"Mismatch: {a_sym} Passing Advantage")
                        if as_['off_rush_rank'] < 10 and hs['def_rush_rank'] > 20:
                            st.info(f"Mismatch: {a_sym} Rushing Advantage")
                    
                    with scol2:
                        st.markdown(f"**{h_sym} Offense vs {a_sym} Defense**")
                        st.markdown(f"- Pass: Rank #{int(hs['off_pass_rank'])} vs #{int(as_['def_pass_rank'])}")
                        st.markdown(f"- Rush: Rank #{int(hs['off_rush_rank'])} vs #{int(as_['def_rush_rank'])}")
                        
                        if hs['off_pass_rank'] < 10 and as_['def_pass_rank'] > 20:
                            st.info(f"Mismatch: {h_sym} Passing Advantage")
                        if hs['off_rush_rank'] < 10 and as_['def_rush_rank'] > 20:
                            st.info(f"Mismatch: {h_sym} Ground Game Advantage")
                    
                    st.divider()
                    
                    # 3. Comparative Graph
                    st.subheader("Statistical Comparison (Strength Score)")
                    categories = ['Passing Off', 'Rushing Off', 'Pass Def', 'Rush Def']
                    # Convert Rank to Score (33 - Rank) so higher bar is better
                    h_vals = [33 - hs['off_pass_rank'], 33 - hs['off_rush_rank'], 33 - hs['def_pass_rank'], 33 - hs['def_rush_rank']]
                    a_vals = [33 - as_['off_pass_rank'], 33 - as_['off_rush_rank'], 33 - as_['def_pass_rank'], 33 - as_['def_rush_rank']]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    # Style
                    fig.patch.set_alpha(0.0) # Transparent background
                    ax.set_facecolor('#1e1e1e')
                    
                    rects1 = ax.bar(x - width/2, a_vals, width, label=a_sym, color='#e73c7e')
                    rects2 = ax.bar(x + width/2, h_vals, width, label=h_sym, color='#23a6d5')
                    
                    ax.set_ylabel('Strength (Higher is Better)', color='white')
                    ax.set_title('Team Strength Comparison', color='white')
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories, color='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.legend(facecolor='#1e1e1e', labelcolor='white')
                    
                    # Remove borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')

                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.divider()

                    # 4. Key Players (Roster)
                    st.subheader("Key Players (Season Leaders)")
                    
                    # Fetch leaders
                    h_leaders = get_top_players(row['season'], h_sym)
                    a_leaders = get_top_players(row['season'], a_sym)
                    
                    # Function to display a single team's leaders
                    def display_team_leaders(team_name, leaders):
                        st.markdown(f"**{team_name} Top Performers**")
                        if not leaders:
                            st.caption("No data.")
                            return

                        # Create 3 sub-columns for Pass/Rush/Rec
                        cols = st.columns(3)
                        categories = ['Passing', 'Rushing', 'Receiving']
                        
                        for i, cat in enumerate(categories):
                            with cols[i]:
                                if cat in leaders:
                                    p = leaders[cat]
                                    if p['photo']:
                                        st.image(p['photo'], width=80)
                                    st.caption(f"**{cat}**")
                                    st.write(f"**{p['name']}**")
                                    st.write(f"_{p['details']}_")
                                else:
                                    st.write(f"No {cat} leader")

                    # Display Away Team
                    display_team_leaders(a_sym, a_leaders)
                    st.divider()
                    # Display Home Team
                    display_team_leaders(h_sym, h_leaders)

    st.divider()

# Check if we have scores
if 'home_score' in schedule.columns:
    upcoming = schedule[schedule['home_score'].isna()]
    completed = schedule[~schedule['home_score'].isna()]
else:
    upcoming = schedule
    completed = pd.DataFrame()

mode = st.radio("View", ["Upcoming Games (Analysis)", "Past Performance", "Historical Analysis", "Team Rosters"], horizontal=True)

if mode == "Upcoming Games (Analysis)":
    st.header(f"Upcoming Games & Value Bets ({len(upcoming)})")
    
    if upcoming.empty:
        st.info("No upcoming games found with current data refresh.")
    else:
        pred_df = upcoming.dropna(subset=['spread_line', 'home_moneyline', 'away_moneyline']).copy()
        
        if pred_df.empty:
            st.warning("No upcoming games have betting lines available yet.")
        else:
            X = pred_df[['spread_line', 'home_moneyline', 'away_moneyline']]
            probs = model.predict_proba(X)[:, 1]
            pred_df['Home Win Prob'] = probs
            pred_df['Implied Prob'] = pred_df['home_moneyline'].apply(moneyline_to_prob)
            pred_df['Edge'] = pred_df['Home Win Prob'] - pred_df['Implied Prob']
            pred_df['Value Bet'] = pred_df['Edge'] > 0.05
            
            for index, row in pred_df.iterrows():
                display_game_card(row, show_result=False, stats=season_stats)

elif mode == "Past Performance":
    st.header("Model Performance on Completed Games")
    if completed.empty:
        st.info("No completed games found.")
    else:
        X, y, clean_df = prepare_features(completed) 
        
        if clean_df.empty:
             st.warning("No completed games with sufficient data.")
        else:
            probs = model.predict_proba(X)[:, 1]
            clean_df['Home Win Prob'] = probs
            clean_df['Prediction Correct'] = ((clean_df['Home Win Prob'] > 0.5) == (clean_df['target'] == 1))
            
            accuracy = clean_df['Prediction Correct'].mean()
            correct_count = clean_df['Prediction Correct'].sum()
            total_count = len(clean_df)
            st.metric("Accuracy on Completed Games", f"{accuracy:.1%} ({correct_count}/{total_count})")
            
            for index, row in clean_df.iterrows():
                display_game_card(row, show_result=True, stats=season_stats)

            for index, row in clean_df.iterrows():
                display_game_card(row, show_result=True, stats=season_stats)

elif mode == "Historical Analysis":
    st.header("Historical Model Performance (2014-2025)")
    
    @st.cache_data
    def get_historical_predictions():
        all_years = list(range(2014, 2026))
        all_data = load_data(all_years)
        
        # Filter valid completed games
        if 'home_score' not in all_data.columns:
            return pd.DataFrame()
            
        completed_games = all_data.dropna(subset=['home_score', 'away_score'])
        if completed_games.empty:
            return pd.DataFrame()
            
        try:
            X_hist, y_hist, df_hist = prepare_features(completed_games)
            if df_hist.empty:
                return pd.DataFrame()
                
            probs = model.predict_proba(X_hist)[:, 1]
            preds = (probs > 0.5).astype(int)
            df_hist['Model Correct'] = (preds == y_hist)
            return df_hist
        except Exception:
            return pd.DataFrame()

    with st.spinner("Calculating historical data..."):
        hist_df = get_historical_predictions()
    
    if not hist_df.empty:
        # 1. Year-over-Year Trend
        st.subheader("Year-over-Year Accuracy")
        yearly_acc = hist_df.groupby('season')['Model Correct'].mean().reset_index()
        yearly_acc.columns = ['Season', 'Accuracy']
        st.line_chart(yearly_acc.set_index('Season')['Accuracy'])
        
        # 2. Team-wise performance
        st.subheader("Team-wise Accuracy")
        st.write("How accurately does the model predict games involving each team?")
        
        # We need to melt the dataframe to have one row per team-game
        home_games = hist_df[['home_team', 'Model Correct']].rename(columns={'home_team': 'Team'})
        away_games = hist_df[['away_team', 'Model Correct']].rename(columns={'away_team': 'Team'})
        team_games = pd.concat([home_games, away_games])
        
        team_acc = team_games.groupby('Team')['Model Correct'].agg(['mean', 'count']).reset_index()
        team_acc.columns = ['Team', 'Accuracy', 'Games Played']
        team_acc = team_acc.sort_values('Accuracy', ascending=False)
        
        # Filter mostly active teams (e.g., > 50 games in total history to avoid noise)
        team_acc = team_acc[team_acc['Games Played'] > 50]

        # Bar chart
        st.bar_chart(team_acc.set_index('Team')['Accuracy'])
        
        # Detailed Table
        st.dataframe(team_acc.style.format({'Accuracy': '{:.1%}'}))
    else:
        st.warning("Could not calculate historical accuracy.")

elif mode == "Team Rosters":
    
    col_header, col_year = st.columns([3, 1])
    with col_year:
        roster_year = st.selectbox("Season", list(range(2025, 2013, -1)), index=0) # Default to 2025
        
    with col_header:
        st.header(f"Team Rosters & Stats ({roster_year})")
    
    # Team Selector
    # Get list of teams from team_desc
    teams_list = team_desc['team_abbr'].unique()
    teams_list = sorted(teams_list)
    
    selected_team = st.selectbox("Select Team", teams_list)
    
    # Get Team Details
    team_info = team_desc[team_desc['team_abbr'] == selected_team].iloc[0]
    full_name = team_info['team_name'] if 'team_name' in team_info else selected_team
    conf = team_info['team_conf'] if 'team_conf' in team_info else ""
    div = team_info['team_division'] if 'team_division' in team_info else ""
    
    # Get Logo
    logo_url = get_logo(selected_team, team_desc)
    col1, col2 = st.columns([1, 5])
    with col1:
        if logo_url:
            st.image(logo_url, width=100)
    with col2:
        st.title(full_name)
        if conf and div:
            st.caption(f"{conf} - {div}")
    
    # Get Stats & Roster
    stats_df = get_team_roster_stats(roster_year, selected_team)
    full_roster = get_full_roster(roster_year, selected_team)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Full Roster", "Passing Leaders", "Rushing Leaders", "Receiving Leaders"])
    
    # Common photo config
    column_config = {
        "headshot_url": st.column_config.ImageColumn("Player", width="small")
    }
    
    with tab1:
        st.subheader("Complete Team Roster & Stats")
        if full_roster.empty:
            st.info("No roster data available.")
        else:
            # Reorder for display
            display_cols = [
                'headshot_url', 'player_name', 'position', 'jersey_number', 
                'age', 'height', 'weight', 'college', 'years_exp',
                'passing_yards', 'passing_tds', 
                'rushing_yards', 'rushing_tds', 
                'receiving_yards', 'receiving_tds'
            ]
            # Filter matches
            display_cols = [c for c in display_cols if c in full_roster.columns]
            
            st.dataframe(
                full_roster[display_cols],
                column_config={
                    "headshot_url": st.column_config.ImageColumn("Photo", width="small"),
                    "player_name": "Name",
                    "position": "Pos",
                    "jersey_number": "Jersey",
                    "years_exp": st.column_config.NumberColumn("Exp", format="%d"),
                    "age": "Age",
                    "college": "College",
                    "height": "Height",
                    "weight": "Weight",
                    "passing_yards": st.column_config.NumberColumn("Pass Yds", format="%d"),
                    "passing_tds": st.column_config.NumberColumn("Pass TDs", format="%d"),
                    "rushing_yards": st.column_config.NumberColumn("Rush Yds", format="%d"),
                    "rushing_tds": st.column_config.NumberColumn("Rush TDs", format="%d"),
                    "receiving_yards": st.column_config.NumberColumn("Rec Yds", format="%d"),
                    "receiving_tds": st.column_config.NumberColumn("Rec TDs", format="%d")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )

    # Only show stats tabs if stats exist
    if not stats_df.empty:
        with tab2:
            st.subheader("Passing Leaders")
            passers = stats_df[stats_df['passing_yards'] > 0].sort_values('passing_yards', ascending=False)
            pass_cols = ['headshot_url', 'player_name', 'position', 'games', 'passing_yards', 'passing_tds', 'interceptions']
            st.dataframe(
                passers[pass_cols],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
            
        with tab3:
            st.subheader("Rushing Leaders")
            rushers = stats_df[stats_df['rushing_yards'] > 0].sort_values('rushing_yards', ascending=False)
            rush_cols = ['headshot_url', 'player_name', 'position', 'games', 'rushing_yards', 'rushing_tds']
            st.dataframe(
                rushers[rush_cols],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
            
        with tab4:
            st.subheader("Receiving Leaders")
            receivers = stats_df[stats_df['receiving_yards'] > 0].sort_values('receiving_yards', ascending=False)
            rec_cols = ['headshot_url', 'player_name', 'position', 'games', 'receiving_yards', 'receptions', 'receiving_tds']
            st.dataframe(
                receivers[rec_cols],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
    else:
        with tab2: st.caption("No passing data")
        with tab3: st.caption("No rushing data")
        with tab4: st.caption("No receiving data")
