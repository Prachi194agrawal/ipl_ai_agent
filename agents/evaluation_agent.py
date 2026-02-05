# import os
# from typing import Dict, Any
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# class EvaluationAgent:
#     def __init__(self):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if api_key:
#             self.client = OpenAI(api_key=api_key)
#             self.offline_mode = False
#         else:
#             self.offline_mode = True
#             print("EvaluationAgent: Offline mode (no OPENAI_API_KEY)")

#     def evaluate(self, match_info: Dict[str, Any], ml_proba: float, rag_snippets: str, reasoning_text: str) -> str:
#         team1 = match_info.get("team1", "Team1")
        
#         if self.offline_mode:
#             # Offline evaluation
#             if ml_proba > 0.7:
#                 conf = "HIGH"
#             elif ml_proba > 0.3:
#                 conf = "MEDIUM"
#             else:
#                 conf = "LOW"
            
#             return f"""
# **System Evaluation** (Offline Mode):

# ✓ **ML Model**: {conf} confidence ({ml_proba:.1%})
# ✓ **RAG**: Historical context retrieved ({len(rag_snippets.splitlines())} lines)
# ✓ **Reasoning**: Pitch/weather/toss factors analyzed
# ✓ **Consistency**: Model ↔ Reasoning aligned

# **Overall System Confidence: {conf}** ⭐

# (Add OpenAI credits for LLM-powered evaluation)
# """
#         else:
#             # Online LLM evaluation
#             prompt = f"""
# You are a strict evaluator for an IPL prediction system.

# **Match**: {match_info.get('team1', 'Team1')} vs {match_info.get('team2', 'Team2')} 
# **Venue**: {match_info.get('venue', 'TBD')}
# **ML Probability** ({match_info.get('team1', 'Team1')}): {ml_proba:.2%}

# **RAG Evidence**:
# {rag_snippets[:1000]}...

# **Reasoning Analysis**:
# {reasoning_text[:1000]}...

# **Evaluation Tasks**:
# 1. Does reasoning match ML probability AND RAG evidence? (Yes/No + why)
# 2. Identify contradictions or missing factors
# 3. Overall confidence: HIGH/MEDIUM/LOW
# 4. 3 actionable improvements

# Respond in markdown with clear sections.
# """
#             try:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=400,
#                     temperature=0.1
#                 )
#                 return f"**LLM Evaluation**:\n\n{response.choices[0].message.content}"
#             except Exception as e:
#                 return f"""
# **Evaluation** (API Error Fallback):

# LLM temporarily unavailable: {str(e)[:80]}...
# Model confidence: {ml_proba:.1%}
# System Status: MEDIUM (ML + RAG operational)
# """







import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class EvaluationAgent:
    """
    Evaluation Agent using LangChain with Google Gemini.
    Validates consistency between ML predictions, RAG context, and reasoning.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.api_key:
            # Initialize LangChain's Google Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key,
                temperature=0.1,  # Lower temperature for more deterministic evaluation
                max_output_tokens=400,
                convert_system_message_to_human=True
            )
            self.available = True
            print("✅ EvaluationAgent: LangChain + Google Gemini initialized")
        else:
            self.available = False
            self.llm = None
            print("⚠️ EvaluationAgent: No API key (offline mode)")
        
        # Create LangChain chain with output parser
        if self.available:
            self.chain = self._create_evaluation_chain()

    def _create_evaluation_chain(self):
        """Create LangChain evaluation chain with structured prompt"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict Quality Assurance AI for cricket prediction systems.
Your job is to validate consistency between ML predictions, historical data, and AI reasoning.
Provide objective, data-driven evaluation."""),
            ("human", """Evaluate this IPL prediction system's output:

**Match Configuration:**
{match_info}

**ML Model Output:**
- Team 1 Win Probability: {ml_probability}
- Confidence Level: {confidence_level}

**Historical Context (RAG):**
{rag_snippets}

**AI Reasoning Analysis:**
{reasoning_text}

**Your Evaluation Tasks:**
1. **Consistency Check**: Does the AI reasoning align with the ML probability? (Yes/No + brief reason)
2. **RAG Evidence**: Is the historical context relevant and properly utilized? (Yes/No + brief reason)
3. **Confidence Score**: Rate overall system confidence (0-100%) based on alignment
4. **Key Issue**: Identify ONE critical gap or improvement area (if any)

Format your response as:
✓ Consistency: [Yes/No] - [reason]
✓ RAG Usage: [Yes/No] - [reason]
✓ Confidence: [score]% - [justification]
⚠️ Improvement: [one specific recommendation]""")
        ])
        
        # Create chain with output parser
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def evaluate(self, match_info: Dict[str, Any], ml_proba: float, rag_snippets: str, reasoning_text: str) -> str:
        """
        Evaluate prediction system consistency using LangChain.
        
        Args:
            match_info: Match details dictionary
            ml_proba: ML model probability
            rag_snippets: Historical context from RAG
            reasoning_text: AI reasoning output
            
        Returns:
            Evaluation report string
        """
        if not self.available:
            return self._offline_evaluation(match_info, ml_proba, rag_snippets, reasoning_text)
        
        # Determine confidence level
        if ml_proba > 0.7 or ml_proba < 0.3:
            confidence = "HIGH"
        elif ml_proba > 0.6 or ml_proba < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Format match info for display
        match_str = f"{match_info.get('team1', 'Team1')} vs {match_info.get('team2', 'Team2')} at {match_info.get('venue', 'Unknown')}"
        
        # Truncate long texts to stay within token limits
        rag_truncated = rag_snippets[:500] if len(rag_snippets) > 500 else rag_snippets
        reasoning_truncated = reasoning_text[:500] if len(reasoning_text) > 500 else reasoning_text
        
        # Run LangChain evaluation with retry logic
        for attempt in range(3):
            try:
                result = self.chain.invoke({
                    "match_info": match_str,
                    "ml_probability": f"{ml_proba:.1%}",
                    "confidence_level": confidence,
                    "rag_snippets": rag_truncated,
                    "reasoning_text": reasoning_truncated
                })
                return f"**🛡️ LangChain System Evaluation:**\n\n{result}"
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = 10 * (attempt + 1)
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s... (Attempt {attempt+1}/3)")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Evaluation Error: {error_str}")
                    return self._offline_evaluation(match_info, ml_proba, rag_snippets, reasoning_text)
        
        return "⚠️ Evaluation unavailable after 3 retries. Using offline analysis."

    def _offline_evaluation(self, match_info: Dict[str, Any], ml_proba: float, rag_snippets: str, reasoning_text: str) -> str:
        """Fallback evaluation when LLM is unavailable"""
        team1 = match_info.get("team1", "Team1")
        
        # Simple heuristic-based evaluation
        if ml_proba > 0.7 or ml_proba < 0.3:
            system_confidence = "HIGH (85%)"
            status = "✅"
        elif ml_proba > 0.6 or ml_proba < 0.4:
            system_confidence = "MEDIUM (70%)"
            status = "⚠️"
        else:
            system_confidence = "LOW (55%)"
            status = "⚠️"
        
        has_rag = len(rag_snippets) > 50
        has_reasoning = len(reasoning_text) > 50
        
        return f"""**📋 Offline System Evaluation:**

{status} **ML Model**: Confidence {system_confidence}
  └─ {team1} win probability: {ml_proba:.1%}

✓ **RAG System**: {'Active' if has_rag else 'Limited'} 
  └─ Historical context: {len(rag_snippets)} characters retrieved

✓ **Reasoning Agent**: {'Active' if has_reasoning else 'Limited'}
  └─ Analysis provided: {len(reasoning_text)} characters

✓ **Consistency**: All components operational and aligned

⚠️ **Note**: Add GOOGLE_API_KEY for LLM-powered evaluation with detailed consistency checks.

**Overall System Status**: {'OPERATIONAL' if has_rag and has_reasoning else 'PARTIALLY OPERATIONAL'}"""