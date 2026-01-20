# NFL Game Outcome Prediction System 🏈

A production-grade sports analytics dashboard that predicts NFL game outcomes using historical data and machine learning. This application provides real-time value bet analysis, team strategy insights, and professional visualizations.

![NFL Predictor Demo](https://via.placeholder.com/800x400?text=NFL+Prediction+Dashboard+Screenshot)

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
    git clone https://github.com/MohithS04/nfl-predictor.git
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

## 📊 Model Details

The system uses a **Logistic Regression** classifier.
*   **Features**: Pre-game betting lines (Spread, Moneyline) and home/away context.
*   **Training Data**: 2014 - Present.
*   **Performance**: ~66-67% accuracy on historical test sets, competitive with market baselines.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements.

## 📝 License

This project is open-source and available for educational purposes.
