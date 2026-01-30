import os
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ReasoningAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.offline_mode = False
        else:
            self.offline_mode = True
            print("ReasoningAgent: Offline mode (no OPENAI_API_KEY)")

    def explain_prediction(self, match_info: Dict[str, Any], ml_proba: float, data_context: Dict[str, Any]) -> str:
        team1 = match_info["team1"]
        team2 = match_info["team2"]
        
        # Determine confidence level
        if ml_proba > 0.6:
            confidence = "High"
        elif ml_proba > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Extract context data safely
        pitch_type = data_context['pitch_report'].get('pitch_type', 'balanced')
        weather_forecast = data_context['weather'].get('forecast', 'Clear')
        toss_winner = match_info.get('toss_winner', 'TBD')
        toss_decision = match_info.get('toss_decision', 'TBD')
        
        if self.offline_mode:
            return f"""
**Expert Analysis** (Offline Mode):

Model predicts **{team1} win probability: {ml_proba:.1%}** ({confidence} confidence)

**Key Factors:**
• **Pitch**: {pitch_type} - favors {pitch_type.split('-')[0] if '-' in pitch_type else pitch_type} teams
• **Weather**: {weather_forecast} - minimal impact expected
• **Toss**: {toss_winner} chose {toss_decision} - slight edge to bowling first
• **Model Insight**: {'Team1 favored' if ml_proba>0.5 else 'Team2 favored'}

(LLM reasoning disabled due to quota. Add OpenAI credits to enable.)
"""
        else:
            # Online LLM reasoning (when quota available)
            prompt = f"""
You are an expert IPL cricket analyst.

Match: {team1} vs {team2}
Venue: {match_info.get('venue', 'TBD')}
ML win probability for {team1}: {ml_proba:.2%}
Pitch: {pitch_type}
Weather: {weather_forecast}
Toss: {toss_winner} chose {toss_decision}

Provide concise expert reasoning:
1. Which team has the edge and why?
2. How do pitch/weather/toss affect prediction?
3. 3 key upset factors.
"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                return f"**LLM Expert Analysis**:\n\n{response.choices[0].message.content}"
            except Exception as e:
                return f"LLM temporarily unavailable: {str(e)[:100]}...\nFalling back to model analysis ({ml_proba:.1%})."


#     def explain_prediction(
#         self,
#         match_info: Dict[str, Any],
#         ml_proba: float,
#         data_context: Dict[str, Any]
#     ) -> str:
#         team1 = match_info["team1"]
#         team2 = match_info["team2"]

#         prompt = f"""
# You are an expert IPL cricket analyst.

# Match: {team1} vs {team2}
# Venue: {match_info['venue']}
# Date: {match_info['date']}
# ML model win probability for {team1}: {ml_proba:.2%}

# Player form: {data_context['player_form']}
# Pitch report: {data_context['pitch_report']}
# Team compositions: {data_context['team_composition']}
# Weather: {data_context['weather']}

# 1. Provide a concise expert reasoning for which team has the edge.
# 2. Explain how pitch, player form, and conditions affect the model prediction.
# 3. Highlight 3 key factors that could cause an upset.
# """
#         response = self.client.responses.create(
#             model="gpt-4.1-mini",  # or chosen model
#             input=prompt,
#             max_output_tokens=400,
#         )
#         return response.output[0].content[0].text
