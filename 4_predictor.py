import pandas as pd
import joblib

# Load trained model and feature data
model = joblib.load('epl_model.joblib')
df_form = pd.read_csv('df_form_all_leagues.csv')

# Convert 'Date' column back to datetime
df_form['Date'] = pd.to_datetime(df_form['Date'])

# Sample upcoming fixtures (edit this block as needed)
upcoming_fixtures = pd.DataFrame({
    'Date': pd.to_datetime(['2025-08-10', '2025-08-11']),
    'HomeTeam': ['Arsenal', 'Chelsea'],
    'AwayTeam': ['Man City', 'Liverpool']
})

# Feature generation function
def generate_features(row, df_form):
    home = row['HomeTeam']
    away = row['AwayTeam']
    date = row['Date']

    # Get latest historical stats before the match date
    home_stats = df_form[(df_form['HomeTeam'] == home) & (df_form['Date'] < date)].sort_values('Date').tail(1)
    away_stats = df_form[(df_form['AwayTeam'] == away) & (df_form['Date'] < date)].sort_values('Date').tail(1)

    if home_stats.empty or away_stats.empty:
        return pd.Series([None] * 6)

    return pd.Series([
        home_stats['AvgGoalsScored_H'].values[0],
        home_stats['AvgGoalsConceded_H'].values[0],
        home_stats['AvgPoints_H'].values[0],
        away_stats['AvgGoalsScored_A'].values[0],
        away_stats['AvgGoalsConceded_A'].values[0],
        away_stats['AvgPoints_A'].values[0],
    ])

# Apply feature generation
upcoming_fixtures[['AvgGoalsScored_H', 'AvgGoalsConceded_H', 'AvgPoints_H',
                   'AvgGoalsScored_A', 'AvgGoalsConceded_A', 'AvgPoints_A']] = upcoming_fixtures.apply(
    lambda row: generate_features(row, df_form), axis=1
)

# Drop rows with missing values
upcoming_fixtures.dropna(inplace=True)

# Predict
features = [
    'AvgGoalsScored_H', 'AvgGoalsConceded_H', 'AvgPoints_H',
    'AvgGoalsScored_A', 'AvgGoalsConceded_A', 'AvgPoints_A'
]
X_pred = upcoming_fixtures[features]
predictions = model.predict(X_pred)

# Map predictions to readable results
label_map = {0: 'AwayWin', 1: 'Draw', 2: 'HomeWin'}
upcoming_fixtures['Prediction'] = predictions
upcoming_fixtures['Prediction'] = upcoming_fixtures['Prediction'].map(label_map)

# Display result
print(upcoming_fixtures[['Date', 'HomeTeam', 'AwayTeam', 'Prediction']])
