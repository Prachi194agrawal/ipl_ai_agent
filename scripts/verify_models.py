#!/usr/bin/env python3
"""
Comprehensive verification of trained models and RAG system
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def check_xgboost_model():
    """Verify XGBoost model exists and can make predictions"""
    print("\n" + "="*60)
    print("🤖 CHECKING XGBOOST MODEL")
    print("="*60)
    
    import joblib
    import numpy as np
    
    model_path = Path("models/ipl_xgb_model.pkl")
    team_enc_path = Path("artifacts/team_encoder.pkl")
    venue_enc_path = Path("artifacts/venue_encoder.pkl")
    
    # Check files exist
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    if not team_enc_path.exists():
        print(f"❌ Team encoder not found: {team_enc_path}")
        return False
    if not venue_enc_path.exists():
        print(f"❌ Venue encoder not found: {venue_enc_path}")
        return False
    
    print(f"✅ Model file exists: {model_path}")
    print(f"✅ Team encoder exists: {team_enc_path}")
    print(f"✅ Venue encoder exists: {venue_enc_path}")
    
    # Load and test
    try:
        model = joblib.load(model_path)
        team_encoder = joblib.load(team_enc_path)
        venue_encoder = joblib.load(venue_enc_path)
        
        print(f"\n📊 Model Info:")
        print(f"   - Teams encoded: {len(team_encoder.classes_)}")
        print(f"   - Venues encoded: {len(venue_encoder.classes_)}")
        print(f"   - Feature count: {model.n_features_in_}")
        
        # Test prediction
        sample_teams = list(team_encoder.classes_[:2])
        sample_venue = venue_encoder.classes_[0]
        
        team1_enc = team_encoder.transform([sample_teams[0]])[0]
        team2_enc = team_encoder.transform([sample_teams[1]])[0]
        venue_enc = venue_encoder.transform([sample_venue])[0]
        toss_winner_enc = team1_enc
        toss_bat = 1
        
        X = np.array([[team1_enc, team2_enc, venue_enc, toss_winner_enc, toss_bat]])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        print(f"\n🧪 Test Prediction:")
        print(f"   Match: {sample_teams[0]} vs {sample_teams[1]}")
        print(f"   Venue: {sample_venue}")
        print(f"   Prediction: {'Team1 Wins' if prediction == 1 else 'Team2 Wins'}")
        print(f"   Confidence: {max(proba):.1%}")
        
        print(f"\n✅ XGBoost model working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_rag_system():
    """Verify RAG system with embeddings"""
    print("\n" + "="*60)
    print("🔍 CHECKING RAG SYSTEM")
    print("="*60)
    
    corpus_path = Path("rag_corpus/matches.jsonl")
    
    if not corpus_path.exists():
        print(f"❌ RAG corpus not found: {corpus_path}")
        return False
    
    print(f"✅ RAG corpus exists: {corpus_path}")
    
    # Count documents
    with open(corpus_path, 'r') as f:
        doc_count = sum(1 for line in f if line.strip())
    print(f"📄 Documents in corpus: {doc_count}")
    
    # Test RAG agent
    try:
        from agents.rag_agent import RAGAgent
        
        print(f"\n🔧 Initializing RAG Agent...")
        rag = RAGAgent(data_path=str(corpus_path))
        
        print(f"\n✅ RAG Agent initialized!")
        print(f"   - Embedding model: all-MiniLM-L6-v2")
        print(f"   - Vector store: FAISS")
        print(f"   - Documents indexed: {len(rag.documents)}")
        
        # Test retrieval
        test_query = "Mumbai Indians wins"
        print(f"\n🧪 Test Retrieval:")
        print(f"   Query: '{test_query}'")
        
        results = rag.retrieve(test_query, k=3)
        print(f"   Retrieved {len(results)} documents:")
        for i, doc in enumerate(results[:2], 1):
            snippet = doc[:80] + "..." if len(doc) > 80 else doc
            print(f"      {i}. {snippet}")
        
        print(f"\n✅ RAG system working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🚀 IPL INSIGHT AGENT - MODEL VERIFICATION")
    print("="*60)
    
    os.chdir(Path(__file__).resolve().parents[1])
    
    xgb_ok = check_xgboost_model()
    rag_ok = check_rag_system()
    
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    print(f"{'✅' if xgb_ok else '❌'} XGBoost Prediction Model")
    print(f"{'✅' if rag_ok else '❌'} RAG System with Embeddings")
    
    if xgb_ok and rag_ok:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n💡 Next steps:")
        print("   1. Run: streamlit run app/streamlit_app.py")
        print("   2. Test predictions in the web interface")
        print("="*60)
        return 0
    else:
        print("\n⚠️ Some systems failed verification")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
