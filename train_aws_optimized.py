#!/usr/bin/env python3
"""
AWS-Optimized Football Predictor
Designed for cloud deployment with professional-grade accuracy
Keeps all advanced features while optimizing memory usage and runtime
"""

import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import gc
from datetime import datetime, timedelta
import psutil
import os
import time

warnings.filterwarnings('ignore')

class AWSOptimizedFootballPredictor:
    def __init__(self, memory_limit_mb=28000):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.start_time = time.time()
        self.memory_limit_mb = memory_limit_mb
        
    def check_memory_limit(self, stage=""):
        """Check if memory limit is exceeded and trigger cleanup if needed."""
        current_memory = self.log_memory_usage(stage)
        if current_memory > self.memory_limit_mb:
            print(f"⚠️  Memory limit exceeded ({current_memory:.1f}MB > {self.memory_limit_mb}MB)")
            print("   Triggering garbage collection...")
            gc.collect()
            return True
        return False
        
    def log_memory_usage(self, stage=""):
        """Monitor memory usage for optimization."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {stage}: {memory_mb:.1f} MB RAM")
        return memory_mb
        
    def load_and_clean_data(self, file_pattern='./e0/*.csv', max_memory_mb=8000):
        """Load and clean football data with memory-constrained batch processing."""
        print("="*60)
        print("🚀 AWS-OPTIMIZED FOOTBALL PREDICTOR (Memory-Constrained)")
        print("="*60)
        
        self.log_memory_usage("Starting data load")
        
        file_paths = glob.glob(file_pattern)
        print(f"Found {len(file_paths)} data files")
        
        # Process files in batches to avoid memory overload
        batch_size = max(1, min(50, len(file_paths) // 4))  # Adaptive batch size
        all_data_chunks = []
        
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}: files {i+1}-{min(i+batch_size, len(file_paths))}")
            
            batch_list = []
            for file in batch_files:
                try:
                    # Try multiple encodings for compatibility
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    df = None
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(file, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if df is not None:
                        batch_list.append(df)
                        
                except Exception as e:
                    print(f"  Skipping {file}: {e}")
            
            if batch_list:
                # Combine batch and immediately clean to save memory
                batch_data = pd.concat(batch_list, ignore_index=True)
                
                # Essential columns for professional model
                columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                          'HTHG', 'HTAG', 'HTR',  # Half-time stats
                          'HS', 'AS', 'HST', 'AST',  # Shots
                          'HF', 'AF', 'HC', 'AC',    # Fouls and Corners
                          'HY', 'AY', 'HR', 'AR']    # Cards
                
                # Add betting odds if available
                betting_cols = ['B365H', 'B365D', 'B365A']
                available_betting = [col for col in betting_cols if col in batch_data.columns]
                if available_betting:
                    columns.extend(available_betting)
                
                # Filter to available columns immediately
                available_columns = [col for col in columns if col in batch_data.columns]
                batch_data = batch_data[available_columns].copy()
                
                all_data_chunks.append(batch_data)
                
                # Clean up batch memory
                del batch_list, batch_data
                gc.collect()
                
                # Check memory usage
                current_memory = self.log_memory_usage(f"Batch {i//batch_size + 1} processed")
                if current_memory > max_memory_mb:
                    print(f"⚠️  Memory limit approached ({current_memory:.1f}MB), processing current data...")
                    break
        
        print(f"✅ Successfully processed {len(all_data_chunks)} batches")
        
        # Combine all batches efficiently
        all_data = pd.concat(all_data_chunks, ignore_index=True)
        
        # Essential columns for professional model
        columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                  'HTHG', 'HTAG', 'HTR',  # Half-time stats
                  'HS', 'AS', 'HST', 'AST',  # Shots
                  'HF', 'AF', 'HC', 'AC',    # Fouls and Corners
                  'HY', 'AY', 'HR', 'AR']    # Cards
        
        # Add betting odds if available (powerful predictors)
        betting_cols = ['B365H', 'B365D', 'B365A']
        available_betting = [col for col in betting_cols if col in all_data.columns]
        if available_betting:
            columns.extend(available_betting)
            print(f"✅ Found betting odds: {available_betting}")
        
        # Filter to available columns
        available_columns = [col for col in columns if col in all_data.columns]
        df_clean = all_data[available_columns].copy()
        
        # Clean up intermediate data
        del all_data, df_list
        gc.collect()
        self.log_memory_usage("After initial cleanup")
        
        # Date processing
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Smart data filtering - use last 4 years for balance of data quantity vs relevance  
        print("📊 Filtering to recent data for relevance...")
        recent_date = df_clean['Date'].max() - timedelta(days=4*365)
        df_clean = df_clean[df_clean['Date'] >= recent_date].reset_index(drop=True)
        print(f"✅ Using {len(df_clean)} matches from last 4 years")
        
        # Memory-efficient data types for numerical columns
        numerical_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                         'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int16')
        
        self.log_memory_usage("Data cleaning complete")
        return df_clean
    
    def create_advanced_features(self, df):
        """Create professional-grade features with memory optimization."""
        print("\n🧠 Creating advanced features...")
        df_form = df.copy()
        
        # Use reasonable sample size for feature creation - balance accuracy vs memory
        if len(df_form) > 6000:
            df_form = df_form.tail(6000).copy()
            print(f"📊 Using most recent {len(df_form)} matches for feature engineering")
            gc.collect()  # Clean memory after sampling
        
        self.log_memory_usage("Starting feature creation")
        
        # ================================
        # BASIC RESULT ENCODING
        # ================================
        df_form['HomeWin'] = df_form['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
        df_form['AwayWin'] = df_form['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))
        
        # ================================
        # OPTIMIZED ROLLING FEATURES
        # ================================
        def create_rolling_features(df, team_col, prefix, windows=[3, 5]):
            """Memory-optimized rolling features with chunking."""
            print(f"  🔄 Rolling features for {prefix} teams...")
            
            for window in windows:
                # Check memory before each window calculation
                if self.check_memory_limit(f"Rolling {prefix} window {window}"):
                    print(f"    Skipping window {window} due to memory constraints")
                    continue
                    
                win_col = 'HomeWin' if prefix == 'H' else 'AwayWin'
                
                # Create rolling stats with memory optimization
                stats_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', win_col]
                available_stats = [col for col in stats_cols if col in df.columns]
                
                if len(available_stats) > 0:
                    # Process teams in chunks to manage memory
                    unique_teams = df[team_col].unique()
                    chunk_size = max(1, len(unique_teams) // 4)
                    
                    rolling_results = []
                    for i in range(0, len(unique_teams), chunk_size):
                        team_chunk = unique_teams[i:i+chunk_size]
                        df_chunk = df[df[team_col].isin(team_chunk)].copy()
                        
                        rolling_chunk = df_chunk.groupby(team_col).rolling(
                            window=window, on='Date', min_periods=1
                        )[available_stats].mean().reset_index()
                        
                        rolling_results.append(rolling_chunk)
                        del df_chunk, rolling_chunk
                        gc.collect()
                    
                    # Combine chunks
                    rolling_stats = pd.concat(rolling_results, ignore_index=True)
                    del rolling_results
                    gc.collect()
                    
                    # Rename columns
                    rename_map = {}
                    for col in available_stats:
                        if col == 'FTHG':
                            rename_map[col] = f'AvgGoalsScored_{prefix}_{window}'
                        elif col == 'FTAG':
                            rename_map[col] = f'AvgGoalsConceded_{prefix}_{window}'
                        elif col == 'HTHG':
                            rename_map[col] = f'AvgHTGoalsScored_{prefix}_{window}'
                        elif col == 'HTAG':
                            rename_map[col] = f'AvgHTGoalsConceded_{prefix}_{window}'
                        elif col == win_col:
                            rename_map[col] = f'AvgPoints_{prefix}_{window}'
                    
                    rolling_stats.rename(columns=rename_map, inplace=True)
                    df = pd.merge(df, rolling_stats, on=[team_col, 'Date'], how='left')
                    
                    del rolling_stats
                    gc.collect()
            
            return df
        
        # Create rolling features for both teams
        df_form = create_rolling_features(df_form, 'HomeTeam', 'H')
        df_form = create_rolling_features(df_form, 'AwayTeam', 'A')
        
        self.log_memory_usage("Rolling features complete")
        
        # ================================
        # DERIVED FEATURES
        # ================================
        print("  📈 Creating derived features...")
        
        for window in [3, 5]:
            # Goal difference (key predictor)
            if f'AvgGoalsScored_H_{window}' in df_form.columns:
                df_form[f'AvgGoalDiff_H_{window}'] = (
                    df_form[f'AvgGoalsScored_H_{window}'] - 
                    df_form[f'AvgGoalsConceded_H_{window}']
                )
            if f'AvgGoalsScored_A_{window}' in df_form.columns:
                df_form[f'AvgGoalDiff_A_{window}'] = (
                    df_form[f'AvgGoalsScored_A_{window}'] - 
                    df_form[f'AvgGoalsConceded_A_{window}']
                )
            
            # Form difference (relative strength)
            if f'AvgPoints_H_{window}' in df_form.columns and f'AvgPoints_A_{window}' in df_form.columns:
                df_form[f'PointsDiff_{window}'] = (
                    df_form[f'AvgPoints_H_{window}'] - df_form[f'AvgPoints_A_{window}']
                )
        
        # ================================
        # BETTING ODDS FEATURES (if available)
        # ================================
        if 'B365H' in df_form.columns:
            print("  💰 Processing betting odds features...")
            # Implied probabilities (market wisdom)
            df_form['ImpliedProb_H'] = 1 / df_form['B365H']
            df_form['ImpliedProb_D'] = 1 / df_form['B365D'] 
            df_form['ImpliedProb_A'] = 1 / df_form['B365A']
            
            # Market efficiency indicators
            df_form['HomeFavorite'] = (df_form['B365H'] < df_form['B365A']).astype(int)
            df_form['DrawLikely'] = (df_form['B365D'] < 3.0).astype(int)
        
        # ================================
        # SIMPLIFIED TEAM STRENGTH (H2H proxy)
        # ================================
        print("  ⚽ Adding team strength indicators...")
        
        # Team strength based on recent performance (replaces complex H2H) - chunked processing
        print("  Processing team strength in chunks...")
        unique_teams = list(set(df_form['HomeTeam'].unique()) | set(df_form['AwayTeam'].unique()))
        chunk_size = max(1, len(unique_teams) // 3)
        
        team_strength_chunks = []
        for i in range(0, len(unique_teams), chunk_size):
            team_chunk = unique_teams[i:i+chunk_size]
            chunk_data = df_form[df_form['HomeTeam'].isin(team_chunk)]
            
            if not chunk_data.empty:
                chunk_strength = chunk_data.groupby('HomeTeam').agg({
                    'FTHG': 'mean',
                    'FTAG': 'mean', 
                    'HomeWin': 'mean'
                }).rename(columns={
                    'FTHG': 'TeamStrength_Goals',
                    'FTAG': 'TeamStrength_Conceded',
                    'HomeWin': 'TeamStrength_Points'
                })
                team_strength_chunks.append(chunk_strength)
            
            del chunk_data
            gc.collect()
        
        # Combine team strength chunks
        team_strength = pd.concat(team_strength_chunks, ignore_index=False) if team_strength_chunks else pd.DataFrame()
        del team_strength_chunks
        gc.collect()
        
        if not team_strength.empty:
            df_form = df_form.merge(team_strength, left_on='HomeTeam', right_index=True, how='left', suffixes=('', '_H'))
            df_form = df_form.merge(team_strength, left_on='AwayTeam', right_index=True, how='left', suffixes=('', '_A'))
            del team_strength
            gc.collect()
        
        # Clean up column names
        strength_cols = [col for col in df_form.columns if 'TeamStrength' in col]
        rename_map = {}
        for col in strength_cols:
            if col.endswith('_H'):
                rename_map[col] = col.replace('_H', '_Home')
            elif col.endswith('_A'):
                rename_map[col] = col.replace('_A', '_Away')
        df_form.rename(columns=rename_map, inplace=True)
        
        # Target variable
        df_form['Result'] = df_form['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        
        self.log_memory_usage("Feature creation complete")
        print(f"✅ Created {len(df_form.columns)} total features")
        
        return df_form
    
    def select_best_features(self, df_form):
        """Select optimal features for cloud training."""
        print("\n🎯 Selecting optimal features...")
        
        feature_candidates = []
        
        # Core rolling features
        for window in [3, 5]:
            feature_candidates.extend([
                f'AvgGoalsScored_H_{window}', f'AvgGoalsConceded_H_{window}', f'AvgPoints_H_{window}',
                f'AvgGoalsScored_A_{window}', f'AvgGoalsConceded_A_{window}', f'AvgPoints_A_{window}',
                f'AvgGoalDiff_H_{window}', f'AvgGoalDiff_A_{window}', f'PointsDiff_{window}'
            ])
        
        # Betting odds (if available)
        if 'ImpliedProb_H' in df_form.columns:
            feature_candidates.extend([
                'ImpliedProb_H', 'ImpliedProb_D', 'ImpliedProb_A',
                'HomeFavorite', 'DrawLikely'
            ])
        
        # Team strength indicators
        strength_features = [col for col in df_form.columns if 'TeamStrength' in col]
        feature_candidates.extend(strength_features)
        
        # Filter to existing features
        existing_features = [f for f in feature_candidates if f in df_form.columns]
        
        print(f"✅ Selected {len(existing_features)} features for training")
        self.feature_names = existing_features
        return existing_features
    
    def train_optimized_model(self, df_form, optimize_hyperparameters=True, max_samples=4000):
        """Train model with memory-constrained settings."""
        print("\n🚀 Training memory-optimized model...")
        
        features = self.select_best_features(df_form)
        
        # Prepare data with robust handling and sampling
        model_data = df_form.dropna(subset=features + ['Result'])
        
        # Limit training data size for memory efficiency
        if len(model_data) > max_samples:
            print(f"📊 Sampling {max_samples} most recent matches for memory efficiency")
            model_data = model_data.tail(max_samples).copy()
            gc.collect()
        
        X = model_data[features].fillna(0)  # Handle any remaining NaNs
        y = model_data['Result']
        
        print(f"📊 Training data: {X.shape[0]} matches, {X.shape[1]} features")
        print(f"📊 Class distribution: {Counter(y)}")
        
        if len(X) < 100:
            print("❌ ERROR: Insufficient training data!")
            return None, None, None
        
        # Clean up intermediate data
        del model_data
        gc.collect()
        
        self.log_memory_usage("Data preparation complete")
        
        # Check memory before proceeding
        if self.check_memory_limit("Before model training"):
            print("⚠️  Reducing training complexity due to memory constraints...")
            optimize_hyperparameters = False  # Skip optimization if memory is tight
        
        # Balanced class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"⚖️  Class weights: {class_weight_dict}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.log_memory_usage("Data scaling complete")
        
        if optimize_hyperparameters:
            print("🔧 Optimizing hyperparameters (cloud-efficient)...")
            
            # Memory-constrained parameter grids (reduced search space)
            n_cores = min(4, os.cpu_count() or 4)  # Limit cores to prevent overload
            print(f"🔧 Using {n_cores} cores for training (memory-optimized)")
            
            models_to_try = {
                'RandomForest': {
                    'model': RandomForestClassifier(
                        random_state=42, 
                        class_weight=class_weight_dict, 
                        n_jobs=n_cores,  # Limited cores
                        warm_start=True  # Memory optimization
                    ),
                    'params': {
                        'n_estimators': [150, 200],  # Reduced from [200, 300]
                        'max_depth': [12, 15],       # Reduced from [15, 20]
                        'min_samples_split': [5],    # Reduced options
                        'max_features': ['sqrt']
                    }
                },
                'GradientBoosting': {
                    'model': GradientBoostingClassifier(
                        random_state=42,
                        warm_start=True  # Memory optimization
                    ),
                    'params': {
                        'n_estimators': [150],       # Reduced from [200]
                        'learning_rate': [0.1],      # Reduced options
                        'max_depth': [4],            # Single value
                        'min_samples_split': [5]     # Single value
                    }
                }
            }
            
            best_score = 0
            best_model = None
            best_model_name = ""
            
            for model_name, model_config in models_to_try.items():
                print(f"  🧪 Testing {model_name}...")
                
                grid_search = GridSearchCV(
                    model_config['model'], 
                    model_config['params'],
                    cv=3, 
                    scoring='accuracy',
                    n_jobs=1,      # Sequential processing to avoid memory overload
                    verbose=0,
                    pre_dispatch='1*n_jobs'  # Limit memory by dispatching jobs sequentially
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_model_name = model_name
                    print(f"    ✅ New best: {model_name} = {best_score:.4f}")
                else:
                    print(f"    📊 Score: {grid_search.best_score_:.4f}")
            
            self.model = best_model
            print(f"🏆 Best model: {best_model_name} (CV Score: {best_score:.4f})")
            
        else:
            # Default memory-optimized model
            n_cores = min(4, os.cpu_count() or 4)
            self.model = RandomForestClassifier(
                n_estimators=200,         # Reduced from 300
                max_depth=15,             # Reduced from 20
                min_samples_split=5,      # Increased for regularization
                max_features='sqrt',
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=n_cores,          # Limited cores
                warm_start=True          # Memory optimization
            )
            self.model.fit(X_train_scaled, y_train)
        
        self.log_memory_usage("Model training complete")
        
        # Model evaluation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 FINAL MODEL PERFORMANCE")
        print(f"🎯 Test Accuracy: {accuracy:.1%}")
        print("\n📊 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return X_test, y_test, y_pred
    
    def save_model(self, model_path='aws_optimized_epl_model.joblib', data_path='aws_optimized_df_form.csv', df_form=None):
        """Save the optimized model for production use."""
        print(f"\n💾 Saving model...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'total_runtime': time.time() - self.start_time
        }
        
        joblib.dump(model_data, model_path)
        
        if df_form is not None:
            df_form.to_csv(data_path, index=False)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Data saved to: {data_path}")
        
        total_time = time.time() - self.start_time
        print(f"⏱️  Total runtime: {total_time/60:.1f} minutes")
        self.log_memory_usage("Final memory usage")

def main():
    print("🚀 Starting AWS-Optimized Football Predictor Training")
    
    predictor = AWSOptimizedFootballPredictor()
    
    try:
        # Load and process data
        df_clean = predictor.load_and_clean_data()
        df_form = predictor.create_advanced_features(df_clean)
        
        # Train optimized model
        results = predictor.train_optimized_model(df_form, optimize_hyperparameters=True)
        
        if results[0] is not None:
            # Save model
            predictor.save_model(df_form=df_form)
            
            print("\n" + "="*60)
            print("🎉 AWS-OPTIMIZED MODEL TRAINING COMPLETED!")
            print("🎉 Professional-grade model ready for production!")
            print("="*60)
        else:
            print("❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()