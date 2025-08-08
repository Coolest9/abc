import pandas as pd
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load all .csv files
file_paths = glob.glob('/content/*.csv')
df_list = []
for file in file_paths:
    try:
        df = pd.read_csv(file, encoding='utf-8')
        df_list.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")
all_data = pd.concat(df_list, ignore_index=True)

# Clean and prepare
columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
           'HS', 'HST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
df_clean = all_data[columns].copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean = df_clean.dropna(subset=['Date'])

df_form = df_clean.copy()
df_form = df_form.tail(3000).copy()

# Rolling stats for home team
df_form['HomeWin'] = df_form['FTR'].apply(lambda x: 1 if x == 'H' else (0.5 if x == 'D' else 0))
rolling_home = df_form.groupby('HomeTeam').rolling(window=5, on='Date')[
    ['FTHG', 'FTAG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR', 'HomeWin']
].mean().reset_index()
rolling_home.rename(columns={
    'FTHG': 'AvgGoalsScored_H', 'FTAG': 'AvgGoalsConceded_H',
    'HS': 'AvgShots_H', 'HST': 'AvgShotsOnTarget_H', 'HF': 'AvgFouls_H',
    'HC': 'AvgCorners_H', 'HY': 'AvgYellows_H', 'HR': 'AvgReds_H',
    'HomeWin': 'AvgPoints_H'
}, inplace=True)
df_form = pd.merge(df_form, rolling_home, on=['HomeTeam', 'Date'], how='left')

# Rolling stats for away team
df_form['AwayWin'] = df_form['FTR'].apply(lambda x: 1 if x == 'A' else (0.5 if x == 'D' else 0))
rolling_away = df_form.groupby('AwayTeam').rolling(window=5, on='Date')[
    ['FTAG', 'FTHG', 'AF', 'AC', 'AY', 'AR', 'AwayWin']
].mean().reset_index()
rolling_away.rename(columns={
    'FTAG': 'AvgGoalsScored_A', 'FTHG': 'AvgGoalsConceded_A',
    'AF': 'AvgFouls_A', 'AC': 'AvgCorners_A', 'AY': 'AvgYellows_A',
    'AR': 'AvgReds_A', 'AwayWin': 'AvgPoints_A'
}, inplace=True)
df_form = pd.merge(df_form, rolling_away, on=['AwayTeam', 'Date'], how='left')

# Label results
df_form['Result'] = df_form['FTR'].map({'A': 0, 'D': 1, 'H': 2})

# Extract features
features = [
    'AvgGoalsScored_H', 'AvgGoalsConceded_H', 'AvgPoints_H',
    'AvgGoalsScored_A', 'AvgGoalsConceded_A', 'AvgPoints_A'
]
model_data = df_form.dropna(subset=features + ['Result'])
X = model_data[features]
y = model_data['Result']

# Train-test split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
