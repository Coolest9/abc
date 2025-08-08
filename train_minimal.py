#!/usr/bin/env python3
"""
MINIMAL Football Predictor - Designed to actually run without memory crashes
Uses absolute minimum resources while still creating a functional model
"""

import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import gc
warnings.filterwarnings('ignore')

class MinimalFootballPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_clean_data(self, file_pattern='./e0/*.csv'):
        """Load minimal data with aggressive filtering."""
        print("Loading data files (minimal mode)...")
        file_paths = glob.glob(file_pattern)
        df_list = []
        
        # Load only essential files (reduce from 91 to ~20)
        essential_files = [f for f in file_paths if any(x in f for x in ['E0.csv', 'E0(1', 'E0(2', 'E1.csv', 'E1(1'])][:20]
        
        for file in essential_files:
            try:
                encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None:
                    df_list.append(df)
                else:
                    print(f"Skipping {file}: encoding issues")
            except Exception as e:
                print(f"Skipping {file}: {e}")
        
        print(f"Loaded {len(df_list)} files successfully.")
        all_data = pd.concat(df_list, ignore_index=True)
        
        # MINIMAL columns only
        columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        df_clean = all_data[columns].copy()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # AGGRESSIVE filtering - only last 1000 matches
        df_clean = df_clean.tail(1000).reset_index(drop=True)
        print(f"Using minimal dataset: {len(df_clean)} matches")
        
        # Clean up immediately
        del all_data, df_list
        gc.collect()
        
        return df_clean
    
    def create_minimal_features(self, df):
        """Create only essential features to avoid memory issues."""
        print("Creating minimal features...")
        df_form = df.copy()
        
        # Basic result encoding
        df_form['HomeWin'] = df_form['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
        df_form['AwayWin'] = df_form['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
        
        # Simple team averages instead of rolling windows to avoid memory issues
        print("Creating simple team averages...")
        
        # Home team averages
        home_stats = df_form.groupby('HomeTeam').agg({
            'FTHG': 'mean',
            'FTAG': 'mean', 
            'HomeWin': 'mean'
        }).rename(columns={
            'FTHG': 'AvgGoalsScored_H',
            'FTAG': 'AvgGoalsConceded_H',
            'HomeWin': 'AvgPoints_H'
        })
        
        df_form = df_form.merge(home_stats, left_on='HomeTeam', right_index=True, how='left')
        
        # Away team averages  
        away_stats = df_form.groupby('AwayTeam').agg({
            'FTHG': 'mean',
            'FTAG': 'mean',
            'AwayWin': 'mean' 
        }).rename(columns={
            'FTHG': 'AvgGoalsScored_A',
            'FTAG': 'AvgGoalsConceded_A',
            'AwayWin': 'AvgPoints_A'
        })
        
        df_form = df_form.merge(away_stats, left_on='AwayTeam', right_index=True, how='left')
        
        # Simple derived features
        df_form['GoalDiff_H'] = df_form.get('AvgGoalsScored_H', 0) - df_form.get('AvgGoalsConceded_H', 0)
        df_form['GoalDiff_A'] = df_form.get('AvgGoalsScored_A', 0) - df_form.get('AvgGoalsConceded_A', 0)
        df_form['FormDiff'] = df_form.get('AvgPoints_H', 0) - df_form.get('AvgPoints_A', 0)
        
        # Target variable
        df_form['Result'] = df_form['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        
        # Clean up
        gc.collect()
        
        return df_form
    
    def train_minimal_model(self, df_form):
        """Train with minimal features and no optimization."""
        print("Training minimal model...")
        
        # MINIMAL feature set
        features = [
            'AvgGoalsScored_H', 'AvgGoalsConceded_H', 'AvgPoints_H',
            'AvgGoalsScored_A', 'AvgGoalsConceded_A', 'AvgPoints_A',
            'GoalDiff_H', 'GoalDiff_A', 'FormDiff'
        ]
        
        # Filter existing features
        existing_features = [f for f in features if f in df_form.columns]
        self.feature_names = existing_features
        
        print(f"Using {len(existing_features)} features: {existing_features}")
        
        # Prepare data
        model_data = df_form.dropna(subset=existing_features + ['Result'])
        X = model_data[existing_features].fillna(0)
        y = model_data['Result']
        
        print(f"Training data: {X.shape[0]} matches")
        
        if len(X) < 50:
            print("ERROR: Not enough data for training!")
            return None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # SIMPLE Random Forest - no optimization
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced from 200
            max_depth=10,     # Limited depth
            random_state=42,
            n_jobs=1          # Single thread
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Simple evaluation
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred, target_names=['Away', 'Draw', 'Home']))
        
        return X_test, y_test, y_pred
    
    def save_model(self, model_path='minimal_epl_model.joblib', data_path='minimal_df_form.csv', df_form=None):
        """Save the minimal model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, model_path)
        
        if df_form is not None:
            df_form.to_csv(data_path, index=False)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Data saved to: {data_path}")

def main():
    print("="*50)
    print("MINIMAL FOOTBALL PREDICTOR")
    print("Designed to run on limited memory systems")
    print("="*50)
    
    predictor = MinimalFootballPredictor()
    
    try:
        # Load minimal data
        df_clean = predictor.load_and_clean_data()
        
        # Create minimal features
        df_form = predictor.create_minimal_features(df_clean)
        
        # Train minimal model
        results = predictor.train_minimal_model(df_form)
        
        if results[0] is not None:
            # Save model
            predictor.save_model(df_form=df_form)
            
            print("\n" + "="*50)
            print("✅ MINIMAL MODEL TRAINING COMPLETED!")
            print("✅ Model ready for use with ui.py")
            print("="*50)
        else:
            print("❌ Training failed - insufficient data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()