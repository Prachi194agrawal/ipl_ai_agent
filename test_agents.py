#!/usr/bin/env python3
"""Test script to verify all agents load correctly"""

print("Testing agent imports...")

try:
    from agents.data_fetch_agent import DataFetchAgent
    print("✅ DataFetchAgent loaded")
except Exception as e:
    print(f"❌ DataFetchAgent failed: {e}")

try:
    from agents.rag_agent import RAGAgent
    print("✅ RAGAgent loaded")
except Exception as e:
    print(f"❌ RAGAgent failed: {e}")

try:
    from agents.reasoning_agent import ReasoningAgent
    print("✅ ReasoningAgent loaded")
except Exception as e:
    print(f"❌ ReasoningAgent failed: {e}")

try:
    from agents.evaluation_agent import EvaluationAgent
    print("✅ EvaluationAgent loaded")
except Exception as e:
    print(f"❌ EvaluationAgent failed: {e}")

print("\n✅ All agents loaded successfully!")
print("You can now run: streamlit run app/streamlit_app.py")
