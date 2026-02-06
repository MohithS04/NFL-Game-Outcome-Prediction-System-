# NFL Game Outcome Prediction System 🏈

A production-grade sports analytics dashboard that predicts NFL game outcomes using historical data and machine learning. This application provides real-time value bet analysis, team strategy insights, and professional visualizations.


![NFL Predictor Demo](https://via.placeholder.com/800x400?text=NFL+Prediction+Dashboard+Screenshot)

## 📖 About the Project

This project is a comprehensive sports analytics platform designed to bridge the gap between raw data and actionable betting insights. By leveraging over a decade of historical NFL data and integrating real-time live scores, the application serves as a "smart assistant" for analyzing matchups.

The core philosophy is simple: **Value Detection**. Instead of just predicting who will win, the system compares its internal calculated probabilities against the compiled betting market odds (Implied Probability). When the model's confidence significantly exceeds the market's, it flags a "Value Bet"—identifying opportunities where the market may be inefficient.

### 🎯 Project Objectives
*   **End-to-End ML Pipeline**: From raw data ingestion (`nfl_data_py`) to feature engineering, model training, and deployment.
*   **Real-Time Integration**: Seamlessly blending historical stats with live game data from the **ESPN API**.
*   **Professional UX**: Moving beyond basic charts to a modern, glassmorphism-inspired interface that feels high-end.

## 🔬 Methodology & Data

### 1. Data Sources
*   **Historical Data (2014-Present)**: Sourced via `nfl_data_py`, providing deep access to play-by-play, schedules, and roster data.
*   **Live Data**: Real-time scores and game status fetched directly from the **ESPN Public API**, allowing the dashboard to "go live" on game day.

### 2. Target Variable
The model treats every game as a binary classification problem:
*   **Target**: **Home Team Win**
*   **Label**: `1` (Home Win), `0` (Away Win/Tie)
*   **Output**: A calibrated probability (0% to 100%) representing the likelihood of a Home victory.

### 3. Feature Engineering
The model prioritizes **market efficiency signals** combined with team context:
*   **Betting Lines**: Spread and Moneyline are used as primary features, as they represent the "wisdom of the crowd."
*   **Context**: Home field advantage is implicitly baked into the model's training data.
*   **Team Strength (EPA)**: For the "Matchup Strategy" view, we calculate **Expected Points Added (EPA)** per play to rank Offenses and Defenses, identifying specific tactical mismatches (e.g., "Top Tier Passing Offense vs. Bottom Tier Pass Defense").

### 4. Model Architecture
We utilize a **Logistic Regression** classifier. While simple, it provides superior **probability calibration** compared to complex tree-based models for sports outcomes, where noise is high and interpretability is key.

## 🚀 Features

*   **Machine Learning Model**: Logistic Regression model trained on 10+ years of NFL data (2014-2025) to predict win probabilities.
*   **Real-time Analysis**:
    *   **Value Bets**: Automatically highlights "Edge" opportunities where the model's probability exceeds market implied probability.
    *   **Win Ratio Visuals**: Visual probability bars showing the split between Home and Away teams.
*   **Deep Strategic Insights**:
    *   **Team Fitness**: Analyzing recent form (last 5 games) with win/loss records.
    *   **Matchup Strategy**: Detailed breakdown of Offensive vs Defensive rankings (Passing & Rushing).
    *   **Key Mismatches**: Automatic detection of strategic advantages (e.g., "Passing Attack Dominance").
    *   **Comparative Graphs**: Side-by-side strength comparison charts.
*   **Historical Performance Tracking**:
    *   **Year-over-Year Accuracy**: Line charts tracking model performance over time.
    *   **Team-wise Analysis**: Detailed breakdown of accuracy per team.
*   **Professional UI/UX**:
    *   Animated "Football Field" background.
    *   Glassmorphism design for a modern, sleek aesthetic.
    *   Official Team Logos integrated throughout.

## 🛠️ Tech Stack

*   **Python 3.8+**
*   **Streamlit**: For the interactive web dashboard.
*   **Scikit-Learn**: For machine learning modeling.
*   **Pandas & NumPy**: For data manipulation and feature engineering.
*   **nfl_data_py**: For fetching comprehensive NFL play-by-play and schedule data.
*   **Matplotlib & Seaborn**: For statistical visualizations.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/nfl-predictor.git
    cd nfl-predictor
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

You can run the entire system using the provided helper script:

```bash
./run.sh
```

Alternatively, run the components individually:

1.  **Train the Model** (Optional if `model.pkl` exists)
    ```bash
    python3 -m src.model
    ```

2.  **Launch the Dashboard**
    ```bash
    streamlit run app.py
    ```

Access the app at `http://localhost:8501`.

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── run.sh                 # Helper script to train & run
├── model.pkl              # Trained model file
├── src/
│   ├── data.py            # Data fetching & caching logic
│   ├── features.py        # Feature engineering pipeline
│   └── model.py           # Model training & evaluation script
└── README.md              # Project documentation
```


## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

## 📝 License

This project is open-source and available for educational purposes.
