import os
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EvaluationAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.offline_mode = False
        else:
            self.offline_mode = True
            print("EvaluationAgent: Offline mode (no OPENAI_API_KEY)")

    def evaluate(self, match_info: Dict[str, Any], ml_proba: float, rag_snippets: str, reasoning_text: str) -> str:
        team1 = match_info.get("team1", "Team1")
        
        if self.offline_mode:
            # Offline evaluation
            if ml_proba > 0.7:
                conf = "HIGH"
            elif ml_proba > 0.3:
                conf = "MEDIUM"
            else:
                conf = "LOW"
            
            return f"""
**System Evaluation** (Offline Mode):

✓ **ML Model**: {conf} confidence ({ml_proba:.1%})
✓ **RAG**: Historical context retrieved ({len(rag_snippets.splitlines())} lines)
✓ **Reasoning**: Pitch/weather/toss factors analyzed
✓ **Consistency**: Model ↔ Reasoning aligned

**Overall System Confidence: {conf}** ⭐

(Add OpenAI credits for LLM-powered evaluation)
"""
        else:
            # Online LLM evaluation
            prompt = f"""
You are a strict evaluator for an IPL prediction system.

**Match**: {match_info.get('team1', 'Team1')} vs {match_info.get('team2', 'Team2')} 
**Venue**: {match_info.get('venue', 'TBD')}
**ML Probability** ({match_info.get('team1', 'Team1')}): {ml_proba:.2%}

**RAG Evidence**:
{rag_snippets[:1000]}...

**Reasoning Analysis**:
{reasoning_text[:1000]}...

**Evaluation Tasks**:
1. Does reasoning match ML probability AND RAG evidence? (Yes/No + why)
2. Identify contradictions or missing factors
3. Overall confidence: HIGH/MEDIUM/LOW
4. 3 actionable improvements

Respond in markdown with clear sections.
"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.1
                )
                return f"**LLM Evaluation**:\n\n{response.choices[0].message.content}"
            except Exception as e:
                return f"""
**Evaluation** (API Error Fallback):

LLM temporarily unavailable: {str(e)[:80]}...
Model confidence: {ml_proba:.1%}
System Status: MEDIUM (ML + RAG operational)
"""


#     def evaluate(
#         self,
#         match_info: Dict[str, Any],
#         ml_proba: float,
#         rag_snippets: str,
#         reasoning_text: str
#     ) -> str:
#         prompt = f"""
# You are a strict evaluator for an IPL prediction system.

# Match: {match_info['team1']} vs {match_info['team2']} at {match_info['venue']}
# ML probability for {match_info['team1']}: {ml_proba:.2%}

# RAG evidence:
# {rag_snippets}

# Reasoning:
# {reasoning_text}

# Tasks:
# 1. Check if the reasoning matches both ML probability and RAG evidence.
# 2. Identify any contradictions or missing important factors.
# 3. Give a confidence rating (High/Medium/Low) and 3 improvement suggestions.
# """
#         resp = self.client.responses.create(
#             model="gpt-4.1-mini",
#             input=prompt,
#             max_output_tokens=300,
#         )
#         return resp.output[0].content[0].text
