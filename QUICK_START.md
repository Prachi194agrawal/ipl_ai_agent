# 🚀 Quick Start Guide - IPL Insight Agent

## Get Started in 3 Minutes!

### Prerequisites
- Python 3.8+
- Internet connection (for first-time setup)
- 4GB RAM minimum

---

## Method 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/Prachi194agrawal/ipl_ai_agent.git
cd ipl_ai_agent

# Run automated setup script
chmod +x setup.sh
./setup.sh

# Configure API key
nano .env  # Add GOOGLE_API_KEY

# Start application
streamlit run app/streamlit_app.py
```

**Open browser**: http://localhost:8501

---

## Method 2: Manual Setup

### Step 1: Clone & Setup Environment
```bash
git clone https://github.com/Prachi194agrawal/ipl_ai_agent.git
cd ipl_ai_agent
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: First installation takes 5-10 minutes (downloads HuggingFace models)

### Step 3: Configure API Keys
```bash
cp .env.example .env
nano .env  # or use any text editor
```

Add your Google Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

**Get Free API Key**: https://makersuite.google.com/app/apikey

### Step 4: Train Model (if not included)
```bash
python scripts/train_model.py
```

**Output**: 
- `models/ipl_xgb_model.pkl` (Trained model)
- `artifacts/team_encoder.pkl` (Team encoder)
- `artifacts/venue_encoder.pkl` (Venue encoder)

### Step 5: Build RAG Corpus
```bash
python scripts/build_rag_corpus.py
```

**Output**: `rag_corpus/matches.jsonl` (Historical match data)

### Step 6: Launch Application
```bash
streamlit run app/streamlit_app.py
```

**Access**: http://localhost:8501

---

## Using the Application

### 1. Configure Match Details (Left Sidebar)
- **Team 1**: Select from dropdown (e.g., Mumbai Indians)
- **Team 2**: Select from dropdown (e.g., Chennai Super Kings)
- **Venue**: Choose stadium (e.g., Wankhede Stadium)
- **City**: Enter city name (e.g., Mumbai)
- **Match Date**: Pick date
- **Toss Winner**: Select team that won toss
- **Toss Decision**: Bat or Field

### 2. Ask Expert (Optional)
Type natural language questions:
- "What is the head-to-head record?"
- "How does the pitch affect this match?"
- "What is the historical trend at this venue?"

### 3. Get Prediction
Click **"🚀 Predict & Analyze"** button

### 4. View Results
- **ML Prediction**: Win probabilities with progress bar
- **Historical Context**: RAG-retrieved match data
- **AI Analysis**: LangChain + Gemini expert reasoning
- **System Evaluation**: Consistency validation

---

## Sample Prediction Flow

**Input**:
```
Team 1: Mumbai Indians
Team 2: Chennai Super Kings
Venue: Wankhede Stadium
City: Mumbai
Toss: MI chose Bat
```

**Output**:
```
📊 ML Prediction
   MI Win: 68.5%
   CSK Win: 31.5%

📚 Historical Context (RAG)
   - MI vs CSK at Wankhede: MI leads 15-8
   - Wankhede: Batting-friendly (avg 185)
   - Toss advantage: Batting first wins 55%

🧠 AI Analysis (LangChain + Gemini)
   • MI favored due to superior venue record
   • Pitch conditions suit MI's aggressive batting
   • Dew factor could help CSK if chasing

🛡️ System Evaluation
   ✓ Consistency: Yes - Reasoning aligns with 68% prediction
   ✓ RAG Usage: Yes - Historical data properly utilized
   ✓ Confidence: 85% - Strong system alignment
```

---

## Troubleshooting

### Issue: "Model file not found"
**Solution**:
```bash
python scripts/train_model.py
```

### Issue: "RAG corpus missing"
**Solution**:
```bash
python scripts/build_rag_corpus.py
```

### Issue: "Rate limit exceeded (429)"
**Solution**: Wait 60 seconds. System has automatic retry logic.

### Issue: "Import errors after installation"
**Solution**:
```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

### Issue: "Streamlit not opening"
**Solution**:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

---

## Testing Installation

### Quick Test Commands
```bash
# Test agent imports
python -c "from agents.rag_agent import RAGAgent; print('✅ RAG Agent OK')"
python -c "from agents.reasoning_agent import ReasoningAgent; print('✅ Reasoning Agent OK')"
python -c "from agents.evaluation_agent import EvaluationAgent; print('✅ Evaluation Agent OK')"

# Test model loading
python -c "import joblib; joblib.load('models/ipl_xgb_model.pkl'); print('✅ Model OK')"

# Test Google Gemini API
python check_models.py
```

---

## Performance Tips

### Speed Up First Load
```bash
# Pre-download HuggingFace models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Reduce Memory Usage
Edit `agents/rag_agent.py`:
```python
# Use smaller embedding model
model_name='paraphrase-MiniLM-L3-v2'  # 61MB instead of 90MB
```

### Enable Debug Mode
Create `.env`:
```bash
DEBUG=True
```

---

## Architecture Quick Reference

```
User Input → Data Fetch Agent → ML Model → Prediction
                ↓
           RAG Agent (LangChain + HuggingFace)
                ↓
           Reasoning Agent (LangChain + Gemini)
                ↓
           Evaluation Agent (LangChain + Gemini)
                ↓
           Streamlit UI Display
```

---

## File Structure Quick Reference

```
ipl_insight_agent/
├── agents/              # 4 AI agents (LangChain-powered)
├── app/                 # Streamlit UI
├── data/                # IPL dataset (600+ matches)
├── models/              # Trained XGBoost model
├── artifacts/           # Encoders
├── rag_corpus/          # Historical embeddings
├── scripts/             # Training & corpus building
├── README.md            # Full documentation
├── ARCHITECTURE.md      # System architecture
├── QUICK_START.md       # This file
├── requirements.txt     # Dependencies
└── .env                 # API keys (create from .env.example)
```

---

## Next Steps

1. ✅ **Explore the UI**: Try different team combinations
2. ✅ **Ask Questions**: Use natural language queries
3. ✅ **Read Documentation**: See [README.md](README.md) for deep dive
4. ✅ **Understand Architecture**: Check [ARCHITECTURE.md](ARCHITECTURE.md)
5. ✅ **Customize**: Modify agents for your use case

---

## Getting Help

- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/Prachi194agrawal/ipl_ai_agent/issues)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## API Keys (Free Tier)

### Google Gemini API
- **URL**: https://makersuite.google.com/app/apikey
- **Free Tier**: 60 requests/min, 1500/day
- **Cost**: $0 for free tier
- **Required**: Yes (for reasoning & evaluation)

### OpenWeather API (Optional)
- **URL**: https://openweathermap.org/api
- **Free Tier**: 1000 calls/day
- **Cost**: $0 for free tier
- **Required**: No (uses fallback data)

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with `GOOGLE_API_KEY`
- [ ] ML model trained (`models/ipl_xgb_model.pkl` exists)
- [ ] RAG corpus built (`rag_corpus/matches.jsonl` exists)
- [ ] Application runs (`streamlit run app/streamlit_app.py`)
- [ ] Browser opens at http://localhost:8501
- [ ] Prediction works for sample teams

---

<div align="center">

**🎉 Ready to Predict IPL Matches! 🏏**

*For detailed documentation, see [README.md](README.md)*

</div>
