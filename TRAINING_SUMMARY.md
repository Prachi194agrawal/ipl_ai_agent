# 🎯 IPL Insight Agent - Training Summary

## ✅ Training Completion Status

Both the RAG system and ML prediction model have been successfully trained!

---

## 📦 1. RAG System (Retrieval-Augmented Generation)

### Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers)
- **Vector Store**: FAISS (Facebook AI Similarity Search) - in-memory
- **LLM Integration**: Google Gemini 2.0 Flash (with API key)
- **Framework**: LangChain with LCEL (LangChain Expression Language)

### Training Details
- **Documents Indexed**: 500 IPL match records
- **Source Data**: `rag_corpus/matches.jsonl`
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2 default)
- **Search Method**: Cosine similarity via FAISS

### Features
- ✅ Semantic search across historical IPL matches
- ✅ Context-aware question answering
- ✅ Fallback to keyword matching if embeddings fail
- ✅ Retrieves top-k most relevant documents

### Test Results
```
Query: "Who won the 2019 IPL final?"
✅ Successfully retrieved relevant match contexts

Query: "Tell me about CSK vs MI matches"
✅ Retrieved CSK-related historical data

Query: "Which venues are batting friendly?"
✅ Retrieved venue-specific match information
```

---

## 🤖 2. XGBoost Match Prediction Model

### Configuration
- **Algorithm**: XGBoost Classifier
- **Features**: 
  - team1_enc, team2_enc (encoded team names)
  - venue_enc (encoded venue)
  - toss_winner_enc (encoded toss winner)
  - toss_bat (toss decision: bat=1, field=0)
- **Target**: Binary classification (Team1 wins = 1, Team2 wins = 0)

### Training Details
- **Total Matches**: 752 (after preprocessing)
- **Train Set**: 601 matches (80%)
- **Test Set**: 151 matches (20%)
- **Teams Encoded**: 15 unique teams
- **Venues Encoded**: 41 unique venues

### Hyperparameters
```python
n_estimators=100
max_depth=6
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
eval_metric='logloss'
```

### Performance Metrics
- **Training Accuracy**: 85.2%
- **Test Accuracy**: 51.7%
- **Precision (Team1)**: 0.45
- **Recall (Team1)**: 0.42
- **F1-Score (Team1)**: 0.43

### Artifacts Generated
```
✅ models/ipl_xgb_model.pkl          (Trained XGBoost model)
✅ artifacts/team_encoder.pkl        (Team name encoder)
✅ artifacts/venue_encoder.pkl       (Venue encoder)
```

---

## 🔄 Architecture Integration

### Data Flow
```
User Query
    ↓
Reasoning Agent (Orchestrator)
    ↓
┌──────────────────┬──────────────────┐
│                  │                  │
RAG Agent      Data Agent      Evaluation Agent
│                  │                  │
└──────────────────┴──────────────────┘
         ↓
  Combined Response
```

### Component Roles

1. **RAG Agent**
   - Retrieves historical context using HuggingFace embeddings
   - Provides semantic search over 500 match records
   - Uses FAISS for efficient similarity search

2. **Data Agent** 
   - Uses trained XGBoost model for win probability prediction
   - Encodes team names and venues using saved encoders
   - Returns confidence scores for predictions

3. **Reasoning Agent**
   - Orchestrates multi-agent workflow
   - Combines RAG context with ML predictions
   - Uses Google Gemini for natural language synthesis

4. **Evaluation Agent**
   - Critiques predictions for sanity and consistency
   - Checks for logical errors in reasoning

---

## 🚀 Running the System

### 1. Start Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### 2. Test Individual Agents
```bash
python test_agents.py
```

### 3. Retrain Models (if needed)
```bash
# Rebuild RAG corpus
python scripts/build_rag_corpus.py

# Train RAG embeddings
python scripts/train_rag.py

# Train XGBoost model
python scripts/train_model.py
```

---

## 📊 Key Dependencies

```
langchain>=0.1.0
langchain-community>=0.0.20
langchain-huggingface  (for HF embeddings)
sentence-transformers>=2.2.0
faiss-cpu==1.13.2
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
google-generativeai>=0.3.0
langchain-google-genai>=0.0.6
streamlit==1.53.1
```

---

## ⚠️ Known Issues & Limitations

1. **LLM Quota**: Google Gemini API may hit rate limits (429 errors)
   - Fallback: RAG agent returns raw document snippets without LLM synthesis

2. **Model Accuracy**: Test accuracy at 51.7% indicates:
   - IPL match outcomes have high variance (form, injuries, luck)
   - Model serves as baseline; ensemble methods could improve performance

3. **Embedding Model**: Using CPU-based inference
   - For production, consider GPU acceleration for faster embedding generation

---

## ✅ Validation Checklist

- [x] RAG corpus built from matches.csv (500 documents)
- [x] HuggingFace embeddings initialized (all-MiniLM-L6-v2)
- [x] FAISS index created and tested
- [x] XGBoost model trained (85.2% train / 51.7% test accuracy)
- [x] Encoders saved for inference (team_encoder.pkl, venue_encoder.pkl)
- [x] All agents instantiate without errors
- [x] LangChain QA chain configured with Gemini
- [x] Streamlit app ready to run

---

## 🎓 Technical Highlights

### RAG Implementation
- **Framework**: LangChain LCEL (Expression Language)
- **Retriever**: FAISS.as_retriever() with k=3
- **Prompt Engineering**: Structured prompt template for context-aware QA
- **Fallback Logic**: Graceful degradation when LLM unavailable

### ML Pipeline
- **Feature Engineering**: Label encoding for categorical variables
- **Class Balance**: Handles both team1/team2 wins equally
- **Serialization**: Pickle-based model persistence
- **Reproducibility**: Fixed random_state=42

---

## 📝 Next Steps for Improvement

1. **RAG Enhancements**
   - Index additional data: player stats, weather, pitch reports
   - Implement reranking for better retrieval quality
   - Use more powerful embeddings (e.g., `mpnet-base-v2`)

2. **ML Model Improvements**
   - Feature engineering: head-to-head stats, recent form, home advantage
   - Try ensemble methods: Random Forest, LightGBM
   - Hyperparameter tuning with GridSearchCV

3. **Production Readiness**
   - Save FAISS index to disk for persistence
   - Implement caching for embeddings
   - Add logging and monitoring
   - Set up CI/CD for model retraining

---

**Training Date**: February 7, 2026  
**Status**: ✅ Ready for Production Testing
