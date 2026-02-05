#!/bin/bash
# IPL Insight Agent - Complete Setup Script
# Run this script to set up the entire project from scratch

set -e  # Exit on any error

echo "========================================"
echo "  IPL INSIGHT AGENT - SETUP SCRIPT"
echo "========================================"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $python_version"

# Create virtual environment
echo ""
echo "🏗️  Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"

# Install dependencies
echo ""
echo "📚 Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt 

# Check if .env exists
echo ""
echo "🔑 Checking environment configuration..."
if [ -f ".env" ]; then
    echo "✅ .env file found"
else
    echo "⚠️  .env file not found!"
    echo "   Creating from template..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
    echo "   • GOOGLE_API_KEY (required for reasoning/evaluation)"
    echo "   • OPENWEATHER_API_KEY (optional for weather data)"
fi

# Check if model exists
echo ""
echo "🤖 Checking ML model..."
if [ -f "models/ipl_xgb_model.pkl" ]; then
    echo "✅ ML model found"
else
    echo "⚠️  ML model not found!"
    echo "   Training model now (this may take 2-3 minutes)..."
    python scripts/train_model.py
    echo "✅ Model training complete"
fi

# Check if RAG corpus exists
echo ""
echo "📚 Checking RAG corpus..."
if [ -f "rag_corpus/matches.jsonl" ]; then
    echo "✅ RAG corpus found"
else
    echo "⚠️  RAG corpus not found!"
    echo "   Building RAG corpus..."
    python scripts/build_rag_corpus.py
    echo "✅ RAG corpus built"
fi

# Test imports
echo ""
echo "🧪 Testing agent imports..."
python -c "
try:
    from agents.rag_agent import RAGAgent
    from agents.reasoning_agent import ReasoningAgent
    from agents.evaluation_agent import EvaluationAgent
    from agents.data_fetch_agent import DataFetchAgent
    print('✅ All agents imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Final checks
echo ""
echo "========================================"
echo "  ✅ SETUP COMPLETE!"
echo "========================================"
echo ""
echo "📋 System Status:"
echo "   ✓ Python: $python_version"
echo "   ✓ Virtual Environment: Active"
echo "   ✓ Dependencies: Installed"
echo "   ✓ ML Model: Ready"
echo "   ✓ RAG Corpus: Ready"
echo "   ✓ Agents: Loaded"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Configure API keys (if not done):"
echo "   nano .env"
echo ""
echo "2. Start the application:"
echo "   streamlit run app/streamlit_app.py"
echo ""
echo "3. Open browser to:"
echo "   http://localhost:8501"
echo ""
echo "========================================"
echo "Happy Predicting! 🏏"
echo "========================================"
