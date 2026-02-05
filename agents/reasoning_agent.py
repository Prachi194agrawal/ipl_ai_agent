# import os
# from typing import Dict, Any
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# class ReasoningAgent:
#     def __init__(self):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if api_key:
#             self.client = OpenAI(api_key=api_key)
#             self.offline_mode = False
#         else:
#             self.offline_mode = True
#             print("ReasoningAgent: Offline mode (no OPENAI_API_KEY)")

#     def explain_prediction(self, match_info: Dict[str, Any], ml_proba: float, data_context: Dict[str, Any]) -> str:
#         team1 = match_info["team1"]
#         team2 = match_info["team2"]
        
#         # Determine confidence level
#         if ml_proba > 0.6:
#             confidence = "High"
#         elif ml_proba > 0.4:
#             confidence = "Medium"
#         else:
#             confidence = "Low"
        
#         # Extract context data safely
#         pitch_type = data_context['pitch_report'].get('pitch_type', 'balanced')
#         weather_forecast = data_context['weather'].get('forecast', 'Clear')
#         toss_winner = match_info.get('toss_winner', 'TBD')
#         toss_decision = match_info.get('toss_decision', 'TBD')
        
#         if self.offline_mode:
#             return f"""
# **Expert Analysis** (Offline Mode):

# Model predicts **{team1} win probability: {ml_proba:.1%}** ({confidence} confidence)

# **Key Factors:**
# • **Pitch**: {pitch_type} - favors {pitch_type.split('-')[0] if '-' in pitch_type else pitch_type} teams
# • **Weather**: {weather_forecast} - minimal impact expected
# • **Toss**: {toss_winner} chose {toss_decision} - slight edge to bowling first
# • **Model Insight**: {'Team1 favored' if ml_proba>0.5 else 'Team2 favored'}

# (LLM reasoning disabled due to quota. Add OpenAI credits to enable.)
# """
#         else:
#             # Online LLM reasoning (when quota available)
#             prompt = f"""
# You are an expert IPL cricket analyst.

# Match: {team1} vs {team2}
# Venue: {match_info.get('venue', 'TBD')}
# ML win probability for {team1}: {ml_proba:.2%}
# Pitch: {pitch_type}
# Weather: {weather_forecast}
# Toss: {toss_winner} chose {toss_decision}

# Provide concise expert reasoning:
# 1. Which team has the edge and why?
# 2. How do pitch/weather/toss affect prediction?
# 3. 3 key upset factors.
# """
#             try:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=400,
#                     temperature=0.3
#                 )
#                 return f"**LLM Expert Analysis**:\n\n{response.choices[0].message.content}"
#             except Exception as e:
#                 return f"LLM temporarily unavailable: {str(e)[:100]}...\nFalling back to model analysis ({ml_proba:.1%})."



import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

class ReasoningAgent:
    """
    Reasoning Agent using LangChain with Google Gemini.
    Provides expert cricket analysis based on ML predictions and contextual data.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.api_key:
            # Initialize LangChain's Google Gemini LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key,
                temperature=0.3,
                max_output_tokens=400,
                convert_system_message_to_human=True
            )
            self.available = True
            print("✅ ReasoningAgent: LangChain + Google Gemini initialized")
        else:
            self.available = False
            self.llm = None
            print("⚠️ ReasoningAgent: No API key (offline mode)")

    def explain_prediction(self, match_info: Dict[str, Any], ml_proba: float, data_context: Dict[str, Any]) -> str:
        """
        Generate expert analysis using LangChain prompt template and Google Gemini.
        
        Args:
            match_info: Dictionary with team1, team2, venue, etc.
            ml_proba: ML model's win probability for team1
            data_context: Dictionary with pitch_report, weather, player_form
            
        Returns:
            Expert analysis string
        """
        if not self.available:
            return self._offline_analysis(match_info, ml_proba, data_context)
        
        team1 = match_info.get("team1", "Team1")
        team2 = match_info.get("team2", "Team2")
        venue = match_info.get("venue", "Unknown")
        
        # Extract context safely
        pitch_info = data_context.get('pitch_report', {})
        pitch_type = pitch_info.get('type', 'Balanced')
        avg_score = pitch_info.get('avg_score', 170)
        
        weather_info = data_context.get('weather', {})
        weather_forecast = weather_info.get('forecast', 'Clear')
        temp = weather_info.get('temp_c', 28)
        humidity = weather_info.get('humidity', 60)
        
        toss_winner = match_info.get('toss_winner', 'TBD')
        toss_decision = match_info.get('toss_decision', 'TBD')
        
        # Create LangChain prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert IPL cricket analyst with 15+ years of experience. 
Provide concise, data-driven analysis in exactly 3 bullet points."""),
            ("human", """Analyze this IPL match prediction:

**Match Details:**
- Teams: {team1} vs {team2}
- Venue: {venue}
- ML Win Probability ({team1}): {probability}%

**Context:**
- Pitch Type: {pitch_type} (Avg Score: {avg_score})
- Weather: {weather} | Temp: {temp}°C | Humidity: {humidity}%
- Toss: {toss_winner} chose to {toss_decision}

**Your Task:**
Provide exactly 3 bullet points:
1. Why the model favors the predicted winner (key statistical factors)
2. How pitch/weather/toss impact this prediction
3. ONE critical upset factor that could change the outcome

Keep each point to 1-2 sentences. Be specific and cricket-focused.""")
        ])
        
        # Format the prompt
        formatted_prompt = prompt_template.format_messages(
            team1=team1,
            team2=team2,
            venue=venue,
            probability=f"{ml_proba:.1%}",
            pitch_type=pitch_type,
            avg_score=avg_score,
            weather=weather_forecast,
            temp=temp,
            humidity=humidity,
            toss_winner=toss_winner,
            toss_decision=toss_decision
        )
        
        # Call LLM with retry logic
        for attempt in range(3):
            try:
                response = self.llm.invoke(formatted_prompt)
                return f"**🧠 LangChain AI Analysis:**\n\n{response.content}"
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = 10 * (attempt + 1)
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s... (Attempt {attempt+1}/3)")
                    time.sleep(wait_time)
                else:
                    print(f"❌ LLM Error: {error_str}")
                    return self._offline_analysis(match_info, ml_proba, data_context)
        
        return "⚠️ AI Analysis unavailable after 3 retries. Using fallback analysis."

    def _offline_analysis(self, match_info: Dict[str, Any], ml_proba: float, data_context: Dict[str, Any]) -> str:
        """Fallback analysis when LLM is unavailable"""
        team1 = match_info.get("team1", "Team1")
        team2 = match_info.get("team2", "Team2")
        
        confidence = "HIGH" if ml_proba > 0.65 else "MEDIUM" if ml_proba > 0.45 else "LOW"
        favorite = team1 if ml_proba > 0.5 else team2
        
        pitch_type = data_context.get('pitch_report', {}).get('type', 'Balanced')
        weather = data_context.get('weather', {}).get('forecast', 'Clear')
        
        return f"""**📊 Offline Analysis ({confidence} Confidence):**

• **Model Prediction**: {favorite} favored with {ml_proba:.1%} win probability
• **Pitch Impact**: {pitch_type} conditions may favor {'batsmen' if 'batting' in pitch_type.lower() else 'bowlers'}
• **Weather**: {weather} - {'Minimal impact expected' if 'clear' in weather.lower() else 'Could affect gameplay'}
• **Key Factor**: Toss decision will be crucial at this venue

*Note: LLM analysis unavailable. Add GOOGLE_API_KEY for AI-powered insights.*"""