import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# ================================
# CONFIGURATION
# ================================
class Config:
    MODEL_PATH = 'improved_epl_model.joblib'
    DATA_PATH = 'improved_df_form.csv'
    LABEL_MAP = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    PREDICTION_COLORS = {
        'Home Win': '#28a745',
        'Draw': '#ffc107', 
        'Away Win': '#dc3545'
    }

# ================================
# DATA LOADING AND CACHING
# ================================
@st.cache_data
def load_improved_model():
    """Load the improved model with scaler and feature names."""
    try:
        model_data = joblib.load(Config.MODEL_PATH)
        return model_data['model'], model_data['scaler'], model_data['feature_names']
    except FileNotFoundError:
        st.error(f"Model file '{Config.MODEL_PATH}' not found.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_improved_data():
    """Load the improved historical data."""
    try:
        df = pd.read_csv(Config.DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error(f"Data file '{Config.DATA_PATH}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ================================
# IMPROVED PREDICTION LOGIC
# ================================
class ImprovedMatchPredictor:
    def __init__(self, model, scaler, feature_names, df_form):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.df_form = df_form
    
    def calculate_team_stats(self, team, is_home, match_date, windows=[3, 5, 10]):
        """Calculate comprehensive team statistics."""
        if is_home:
            team_matches = self.df_form[
                (self.df_form['HomeTeam'] == team) & 
                (self.df_form['Date'] < match_date)
            ].sort_values('Date')
        else:
            team_matches = self.df_form[
                (self.df_form['AwayTeam'] == team) & 
                (self.df_form['Date'] < match_date)
            ].sort_values('Date')
        
        if team_matches.empty:
            return None
        
        stats = {}
        prefix = 'H' if is_home else 'A'
        
        for window in windows:
            recent_matches = team_matches.tail(window)
            if len(recent_matches) == 0:
                continue
                
            # Get the most recent available stats for each window
            if is_home:
                # Use home team columns
                for col in self.feature_names:
                    if f'_{prefix}_{window}' in col and col in recent_matches.columns:
                        if not recent_matches[col].isna().all():
                            stats[col] = recent_matches[col].dropna().iloc[-1]
            else:
                # Use away team columns  
                for col in self.feature_names:
                    if f'_{prefix}_{window}' in col and col in recent_matches.columns:
                        if not recent_matches[col].isna().all():
                            stats[col] = recent_matches[col].dropna().iloc[-1]
        
        return stats
    
    def get_h2h_stats(self, home_team, away_team, match_date):
        """Get head-to-head statistics."""
        h2h_matches = self.df_form[
            (((self.df_form['HomeTeam'] == home_team) & (self.df_form['AwayTeam'] == away_team)) |
             ((self.df_form['HomeTeam'] == away_team) & (self.df_form['AwayTeam'] == home_team))) &
            (self.df_form['Date'] < match_date)
        ].sort_values('Date').tail(5)
        
        if len(h2h_matches) >= 1:
            home_wins = len(h2h_matches[
                ((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) |
                ((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
            ])
            draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
            
            return {
                'H2H_HomeWinRate': home_wins / len(h2h_matches),
                'H2H_DrawRate': draws / len(h2h_matches),
                'H2H_AvgTotalGoals': (h2h_matches['FTHG'] + h2h_matches['FTAG']).mean()
            }
        else:
            return {
                'H2H_HomeWinRate': 0.33,
                'H2H_DrawRate': 0.33, 
                'H2H_AvgTotalGoals': 2.5
            }
    
    def generate_features(self, home_team, away_team, match_date):
        """Generate comprehensive features for prediction."""
        features = {}
        
        # Get team stats
        home_stats = self.calculate_team_stats(home_team, True, match_date)
        away_stats = self.calculate_team_stats(away_team, False, match_date)
        
        if home_stats is None or away_stats is None:
            return None
        
        # Combine stats
        features.update(home_stats)
        features.update(away_stats)
        
        # Get head-to-head stats
        h2h_stats = self.get_h2h_stats(home_team, away_team, match_date)
        features.update(h2h_stats)
        
        # Add betting odds if available (use league averages as defaults)
        if 'ImpliedProb_H' in self.feature_names:
            features.update({
                'ImpliedProb_H': 0.45,  # Default home advantage
                'ImpliedProb_D': 0.27,  # Default draw probability
                'ImpliedProb_A': 0.35,  # Default away probability
                'HomeFavorite': 1,      # Assume home favorite by default
                'DrawLikely': 0,        # Default not draw likely
                'BookmakerMargin': 1.07 # Typical bookmaker margin
            })
        
        return features
    
    def predict_match(self, home_team, away_team, match_date):
        """Predict match outcome with improved model."""
        features_dict = self.generate_features(home_team, away_team, match_date)
        if features_dict is None:
            return None
        
        # Create feature vector in correct order
        feature_vector = []
        missing_features = []
        
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                feature_vector.append(features_dict[feature_name])
            else:
                # Use reasonable defaults for missing features
                if 'Goals' in feature_name:
                    feature_vector.append(1.5)
                elif 'Points' in feature_name:
                    feature_vector.append(1.0)
                elif 'Shots' in feature_name:
                    feature_vector.append(10.0)
                elif 'Accuracy' in feature_name or 'Efficiency' in feature_name:
                    feature_vector.append(0.3)
                else:
                    feature_vector.append(0.0)
                missing_features.append(feature_name)
        
        # Scale features
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get feature importance for this prediction (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return {
            'prediction': Config.LABEL_MAP[prediction],
            'probabilities': {
                'Home Win': probabilities[2],
                'Draw': probabilities[1], 
                'Away Win': probabilities[0]
            },
            'features': features_dict,
            'feature_importance': feature_importance,
            'missing_features': missing_features,
            'confidence': max(probabilities)
        }

# ================================
# ENHANCED UI COMPONENTS
# ================================
class EnhancedUIComponents:
    @staticmethod
    def render_header():
        """Render improved header with model info."""
        st.title("⚽ Advanced Football Match Predictor")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Model Type", "Advanced ML")
        with col2:
            st.metric("📊 Features", "40+ Advanced")
        with col3:
            st.metric("🧠 Algorithm", "Optimized RF/GB")
        
        st.markdown("Predict football match outcomes using advanced machine learning with comprehensive team statistics!")
    
    @staticmethod
    def render_prediction_with_confidence(result, home_team, away_team):
        """Enhanced prediction display with confidence indicators."""
        prediction = result['prediction']
        probabilities = result['probabilities']
        confidence = result['confidence']
        
        st.subheader("🔮 Prediction Result")
        
        # Confidence indicator
        confidence_color = "🟢" if confidence > 0.6 else "🟡" if confidence > 0.45 else "🔴"
        st.markdown(f"**Confidence Level:** {confidence_color} {confidence:.1%}")
        
        # Main prediction with enhanced styling
        color = Config.PREDICTION_COLORS.get(prediction, '#6c757d')
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}dd, {color});
                color: white; 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center; 
                font-size: 28px; 
                font-weight: bold;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            ">
                {prediction}
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Enhanced probability visualization
        st.subheader("📊 Probability Analysis")
        
        # Create probability chart
        outcomes = ['Home Win', 'Draw', 'Away Win']
        probs = [probabilities[outcome] for outcome in outcomes]
        colors = [Config.PREDICTION_COLORS[outcome] for outcome in outcomes]
        
        fig = go.Figure(data=[
            go.Bar(
                x=outcomes,
                y=probs,
                marker_color=colors,
                text=[f'{p:.1%}' for p in probs],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Match Outcome Probabilities",
            yaxis_title="Probability",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏠 Home Win", f"{probabilities['Home Win']:.1%}")
        with col2:
            st.metric("🤝 Draw", f"{probabilities['Draw']:.1%}")
        with col3:
            st.metric("✈️ Away Win", f"{probabilities['Away Win']:.1%}")
    
    @staticmethod
    def render_advanced_analytics(result, home_team, away_team):
        """Render advanced analytics and insights."""
        features = result['features']
        
        with st.expander("📈 Advanced Team Analytics", expanded=False):
            # Team comparison radar chart
            st.subheader("Team Performance Comparison")
            
            # Collect comparable metrics
            comparison_metrics = {}
            windows = [5]  # Focus on 5-game form
            
            for window in windows:
                home_goals = features.get(f'AvgGoalsScored_H_{window}', 0)
                away_goals = features.get(f'AvgGoalsScored_A_{window}', 0)
                home_conceded = features.get(f'AvgGoalsConceded_H_{window}', 0)
                away_conceded = features.get(f'AvgGoalsConceded_A_{window}', 0)
                home_points = features.get(f'AvgPoints_H_{window}', 0)
                away_points = features.get(f'AvgPoints_A_{window}', 0)
                
                comparison_metrics = {
                    'Attack': [home_goals, away_goals],
                    'Defense': [3-home_conceded, 3-away_conceded],  # Inverted for better viz
                    'Form': [home_points, away_points],
                    'Goal Difference': [
                        features.get(f'AvgGoalDiff_H_{window}', 0),
                        features.get(f'AvgGoalDiff_A_{window}', 0)
                    ]
                }
            
            # Create comparison chart
            if comparison_metrics:
                metrics_df = pd.DataFrame(comparison_metrics, index=[home_team, away_team])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=metrics_df.loc[home_team].tolist(),
                    theta=list(metrics_df.columns),
                    fill='toself',
                    name=home_team,
                    line_color='#28a745'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=metrics_df.loc[away_team].tolist(),
                    theta=list(metrics_df.columns),
                    fill='toself',
                    name=away_team,
                    line_color='#dc3545'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 3])
                    ),
                    showlegend=True,
                    title="Team Performance Radar"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Head-to-head analysis
        with st.expander("🥊 Head-to-Head Analysis", expanded=False):
            h2h_home_rate = features.get('H2H_HomeWinRate', 0.33)
            h2h_draw_rate = features.get('H2H_DrawRate', 0.33)
            h2h_away_rate = 1 - h2h_home_rate - h2h_draw_rate
            h2h_avg_goals = features.get('H2H_AvgTotalGoals', 2.5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Historical Matchups")
                h2h_fig = go.Figure(data=[
                    go.Pie(
                        labels=[f'{home_team} Wins', 'Draws', f'{away_team} Wins'],
                        values=[h2h_home_rate, h2h_draw_rate, h2h_away_rate],
                        marker_colors=['#28a745', '#ffc107', '#dc3545']
                    )
                ])
                h2h_fig.update_layout(title="Head-to-Head Record")
                st.plotly_chart(h2h_fig, use_container_width=True)
            
            with col2:
                st.subheader("Match Statistics")
                st.metric("Average Goals per Game", f"{h2h_avg_goals:.1f}")
                st.metric(f"{home_team} Win Rate", f"{h2h_home_rate:.1%}")
                st.metric("Draw Rate", f"{h2h_draw_rate:.1%}")
                st.metric(f"{away_team} Win Rate", f"{h2h_away_rate:.1%}")
        
        # Feature importance (if available)
        if result.get('feature_importance'):
            with st.expander("🎯 Key Factors", expanded=False):
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v} 
                    for k, v in result['feature_importance'].items()
                ]).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df, 
                    y='Feature', 
                    x='Importance',
                    orientation='h',
                    title="Top 10 Most Important Factors"
                )
                st.plotly_chart(fig, use_container_width=True)

# ================================
# MAIN APPLICATION
# ================================
def main():
    st.set_page_config(
        page_title="Advanced Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    # Load improved model and data
    model, scaler, feature_names = load_improved_model()
    df_form = load_improved_data()
    
    if None in [model, scaler, feature_names, df_form]:
        st.error("❌ Could not load model or data. Please ensure the improved model files exist.")
        st.info("Run the improved model training script first!")
        st.stop()
    
    # Initialize predictor
    predictor = ImprovedMatchPredictor(model, scaler, feature_names, df_form)
    
    # Header
    EnhancedUIComponents.render_header()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Match Prediction", "📊 Team Analysis", "📈 Model Info", "ℹ️ About"])
    
    with tab1:
        st.header("Advanced Match Prediction")
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            home_teams = sorted(df_form['HomeTeam'].unique())
            home_team = st.selectbox("🏠 Home Team", home_teams)
        
        with col2:
            away_teams = sorted(df_form['AwayTeam'].unique())
            away_team = st.selectbox("✈️ Away Team", away_teams)
        
        # Date selection
        match_date = st.date_input("📅 Match Date", value=date.today())
        match_datetime = datetime.combine(match_date, datetime.min.time())
        
        # Prediction
        if home_team != away_team:
            if st.button("🔮 Generate Advanced Prediction", type="primary"):
                with st.spinner("Analyzing comprehensive team data..."):
                    result = predictor.predict_match(home_team, away_team, match_datetime)
                    
                    if result:
                        EnhancedUIComponents.render_prediction_with_confidence(result, home_team, away_team)
                        EnhancedUIComponents.render_advanced_analytics(result, home_team, away_team)
                        
                        # Show any missing features warning
                        if result['missing_features']:
                            with st.expander("⚠️ Data Limitations", expanded=False):
                                st.warning(f"Some advanced features unavailable: {len(result['missing_features'])} features used defaults")
                    else:
                        st.error("❌ Insufficient data for prediction. Try different teams or check data availability.")
        else:
            st.warning("⚠️ Please select different teams")
    
    with tab2:
        st.header("Team Performance Analysis")
        st.info("🚧 Coming soon: Detailed team performance dashboards, season trends, and comparative analysis")
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Features")
            st.write(f"**Total Features:** {len(feature_names)}")
            st.write("**Key Improvements:**")
            st.write("- Multiple time windows (3, 5, 10 games)")
            st.write("- Advanced derived metrics")
            st.write("- Head-to-head analysis")
            st.write("- Betting odds integration")
            st.write("- Class weight balancing")
        
        with col2:
            st.subheader("🎯 Feature Categories")
            feature_categories = {
                'Goal Statistics': len([f for f in feature_names if 'Goals' in f]),
                'Form & Points': len([f for f in feature_names if 'Points' in f or 'Diff' in f]),
                'Shot Metrics': len([f for f in feature_names if 'Shot' in f]),
                'Advanced Analytics': len([f for f in feature_names if any(x in f for x in ['Efficiency', 'Accuracy', 'Ratio'])]),
                'Head-to-Head': len([f for f in feature_names if 'H2H' in f]),
                'Other': len([f for f in feature_names if not any(x in f for x in ['Goals', 'Points', 'Shot', 'Efficiency', 'Accuracy', 'Ratio', 'H2H'])])
            }
            
            for category, count in feature_categories.items():
                if count > 0:
                    st.metric(category, count)
    
    with tab4:
        st.header("About the Advanced Model")
        st.markdown("""
        ### 🚀 Model Improvements
        
        **Enhanced Feature Engineering:**
        - Multi-window rolling averages (3, 5, 10 games)
        - Advanced derived metrics (efficiency ratios, form differentials)
        - Shot accuracy and defensive solidity metrics
        - Goal scoring and conceding patterns
        
        **Advanced Analytics:**
        - Head-to-head historical performance
        - Betting odds integration (when available)
        - Team strength comparisons
        - Performance trend analysis
        
        **Machine Learning Enhancements:**
        - Class weight balancing for better draw prediction
        - Hyperparameter optimization
        - Cross-validation evaluation
        - Feature importance analysis
        - Support for multiple algorithms (RF, Gradient Boosting)
        
        **Expected Improvements:**
        - Better accuracy on all outcomes
        - Improved draw prediction (previously weakest class)
        - More robust predictions with confidence scoring
        - Better handling of team form and recent performance
        
        ### 📊 How It Works
        The model analyzes each team's recent performance across multiple time windows,
        calculates advanced statistical metrics, considers historical matchups, and
        uses optimized machine learning algorithms to predict match outcomes.
        
        ### ⚠️ Limitations
        - Requires sufficient historical data for both teams
        - Cannot account for injuries, transfers, or other real-time factors
        - Predictions are probabilistic, not guarantees
        - Performance may vary across different leagues and time periods
        """)

if __name__ == "__main__":
    main()