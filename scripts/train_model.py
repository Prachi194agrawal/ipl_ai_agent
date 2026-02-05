#!/usr/bin/env python3
"""
IPL Match Outcome Prediction - Model Training Script
Trains XGBoost classifier on historical IPL match data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    print("✅ Created output directories")

def load_and_preprocess_data(data_path="data/matches.csv"):
    """Load and preprocess IPL match data"""
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Display dataset info
    print(f"✅ Loaded {len(df)} matches")
    print(f"   Columns: {list(df.columns)}")
    
    # Select relevant features
    required_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner']
    df = df[required_cols].dropna()
    
    print(f"✅ After preprocessing: {len(df)} matches")
    return df

def encode_features(df):
    """Encode categorical features"""
    print("🔧 Encoding features...")
    
    # Initialize encoders
    team_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()
    
    # Fit encoders on all unique values
    all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner'], df['winner']]).unique()
    team_encoder.fit(all_teams)
    venue_encoder.fit(df['venue'].unique())
    
    # Encode features
    df['team1_enc'] = team_encoder.transform(df['team1'])
    df['team2_enc'] = team_encoder.transform(df['team2'])
    df['toss_winner_enc'] = team_encoder.transform(df['toss_winner'])
    df['venue_enc'] = venue_encoder.transform(df['venue'])
    df['toss_bat'] = (df['toss_decision'] == 'bat').astype(int)
    
    # Encode target
    df['target'] = (df['winner'] == df['team1']).astype(int)
    
    print(f"✅ Encoded {len(team_encoder.classes_)} teams")
    print(f"✅ Encoded {len(venue_encoder.classes_)} venues")
    
    return df, team_encoder, venue_encoder

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    print("🎯 Training XGBoost model...")
    
    # Initialize model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Test Accuracy: {test_acc:.3f}")
    
    print(f"\n📈 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Team2 Wins', 'Team1 Wins']))
    
    print(f"\n📉 Confusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))
    
    return model

def save_artifacts(model, team_encoder, venue_encoder):
    """Save trained model and encoders"""
    print("\n💾 Saving artifacts...")
    
    joblib.dump(model, "models/ipl_xgb_model.pkl")
    joblib.dump(team_encoder, "artifacts/team_encoder.pkl")
    joblib.dump(venue_encoder, "artifacts/venue_encoder.pkl")
    
    print("✅ Saved models/ipl_xgb_model.pkl")
    print("✅ Saved artifacts/team_encoder.pkl")
    print("✅ Saved artifacts/venue_encoder.pkl")

def main():
    """Main training pipeline"""
    print("="*50)
    print("   IPL MATCH PREDICTION - MODEL TRAINING")
    print("="*50 + "\n")
    
    # Step 1: Setup
    create_directories()
    
    # Step 2: Load data
    df = load_and_preprocess_data()
    
    # Step 3: Encode features
    df_encoded, team_encoder, venue_encoder = encode_features(df)
    
    # Step 4: Prepare train/test split
    print("\n🔀 Splitting data (80/20 train/test)...")
    feature_cols = ['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_bat', 'venue_enc']
    X = df_encoded[feature_cols]
    y = df_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train: {len(X_train)} samples")
    print(f"✅ Test: {len(X_test)} samples")
    
    # Step 5: Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Step 6: Save artifacts
    save_artifacts(model, team_encoder, venue_encoder)
    
    # Step 7: Summary
    print("\n" + "="*50)
    print("   ✅ TRAINING COMPLETE!")
    print("="*50)
    print("\n📦 Generated files:")
    print("   • models/ipl_xgb_model.pkl (XGBoost model)")
    print("   • artifacts/team_encoder.pkl (Team encoder)")
    print("   • artifacts/venue_encoder.pkl (Venue encoder)")
    print("\n🚀 Next steps:")
    print("   1. Run: python scripts/build_rag_corpus.py")
    print("   2. Run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
