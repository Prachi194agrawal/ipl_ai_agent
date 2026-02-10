#!/usr/bin/env python3
"""
RAG System Training and Setup Script
Initializes the RAG agent with HuggingFace embeddings and builds FAISS index
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from agents.rag_agent import RAGAgent
from dotenv import load_dotenv

load_dotenv()

def train_rag_system():
    """Initialize and train RAG system with embeddings"""
    print("=" * 60)
    print("🚀 IPL Insight Agent - RAG System Training")
    print("=" * 60)
    
    # Check if corpus exists
    corpus_path = repo_root / "rag_corpus" / "matches.jsonl"
    if not corpus_path.exists():
        print(f"❌ Error: {corpus_path} not found")
        print("   Run: python scripts/build_rag_corpus.py first")
        return False
    
    print(f"\n✅ Found RAG corpus: {corpus_path}")
    
    # Count documents in corpus
    with open(corpus_path, 'r') as f:
        doc_count = sum(1 for line in f if line.strip())
    print(f"📄 Document count: {doc_count}")
    
    # Initialize RAG Agent (this will automatically build embeddings and index)
    print("\n" + "=" * 60)
    print("🔧 Initializing RAG Agent with HuggingFace Embeddings")
    print("=" * 60)
    
    try:
        rag_agent = RAGAgent(data_path=str(corpus_path))
        print("\n✅ RAG Agent initialized successfully!")
        
        # Test retrieval
        print("\n" + "=" * 60)
        print("🧪 Testing RAG Retrieval")
        print("=" * 60)
        
        test_queries = [
            "Who won the 2019 IPL final?",
            "Tell me about CSK vs MI matches",
            "Which venues are batting friendly?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = rag_agent.retrieve(query, k=3)
            print("📊 Retrieved documents:")
            for i, doc in enumerate(results, 1):
                print(f"   {i}. {doc[:100]}...")
        
        # Test QA if LLM is available
        if rag_agent.qa_chain:
            print("\n" + "=" * 60)
            print("🤖 Testing QA with LLM")
            print("=" * 60)
            
            test_qa = "What do you know about Mumbai Indians?"
            print(f"\n❓ Question: {test_qa}")
            answer = rag_agent.answer_with_context(test_qa)
            print(f"💬 Answer:\n{answer}")
        else:
            print("\n⚠️ LLM not available (set GOOGLE_API_KEY to enable QA)")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ RAG SYSTEM TRAINING COMPLETE")
        print("=" * 60)
        print(f"📦 Embedding Model: all-MiniLM-L6-v2 (HuggingFace)")
        print(f"🗄️ Vector Store: FAISS (in-memory)")
        print(f"📄 Documents Indexed: {doc_count}")
        print(f"🔗 LLM Integration: {'✅ Google Gemini' if rag_agent.qa_chain else '❌ Disabled'}")
        print("\n💡 The RAG agent is ready to use in the Streamlit app!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing RAG Agent: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_rag_system()
    sys.exit(0 if success else 1)
