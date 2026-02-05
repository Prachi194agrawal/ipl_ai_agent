# import requests
# from dataclasses import dataclass
# from typing import Dict, Any, List

# @dataclass
# class MatchContext:
#     team1: str
#     team2: str
#     venue: str
#     city: str
#     date: str  # ISO string

# class DataFetchAgent:
#     def __init__(self):
#         pass

#     def get_player_form(self, team: str) -> Dict[str, Any]:
#         # TODO: integrate proper cricket API; for now use placeholder structure
#         # e.g., read from a local CSV of recent scores
#         return {
#             "team": team,
#             "key_players": [
#                 {"name": "Player A", "recent_runs": 45.3, "recent_wickets": 1.2},
#                 {"name": "Player B", "recent_runs": 36.8, "recent_wickets": 0.4},
#             ]
#         }

#     def get_pitch_report(self, venue: str) -> Dict[str, Any]:
#         # For production, scrape from Cricbuzz/ESPN if allowed.
#         return {
#             "venue": venue,
#             "pitch_type": "batting-friendly",
#             "avg_first_innings_score": 175,
#             "spin_assistance": "moderate"
#         }

#     def get_team_composition(self, team: str) -> Dict[str, Any]:
#         # Read from local JSON of squads for the season.
#         return {
#             "team": team,
#             "probable_xi": [
#                 "Batter1", "Batter2", "Batter3",
#                 "Allrounder1", "Allrounder2",
#                 "Bowler1", "Bowler2", "Bowler3", "Bowler4"
#             ]
#         }

#     def get_weather(self, city: str, date: str) -> Dict[str, Any]:
#         # Integrate with OpenWeatherMap or similar with an API key via env.
#         return {
#             "city": city,
#             "date": date,
#             "forecast": "Clear",
#             "temperature_c": 30,
#             "humidity": 55,
#             "rain_probability": 10
#         }

#     def build_context(self, match: MatchContext) -> Dict[str, Any]:
#         return {
#             "player_form": {
#                 match.team1: self.get_player_form(match.team1),
#                 match.team2: self.get_player_form(match.team2)
#             },
#             "pitch_report": self.get_pitch_report(match.venue),
#             "team_composition": {
#                 match.team1: self.get_team_composition(match.team1),
#                 match.team2: self.get_team_composition(match.team2)
#             },
#             "weather": self.get_weather(match.city, match.date),
#         }



import os
import requests
import random
from dataclasses import dataclass
from typing import Dict, Any, List
from datetime import datetime

@dataclass
class MatchContext:
    team1: str
    team2: str
    venue: str
    city: str
    date: str

class DataFetchAgent:
    def __init__(self):
        self.weather_api_key = os.getenv("OPENWEATHER_API_KEY")

    def get_player_form(self, team: str) -> Dict[str, Any]:
        """Simulates retrieving recent player performance."""
        # In a real app, you would hit a CricAPI here.
        # We simulate realistic data based on the team name.
        return {
            "team": team,
            "key_players": [
                {"name": f"{team} Opener", "recent_runs_avg": 45.3, "strike_rate": 142.5},
                {"name": f"{team} Captain", "recent_runs_avg": 36.8, "strike_rate": 135.0},
                {"name": f"{team} Pacer", "recent_wickets_avg": 1.5, "economy": 8.2},
                {"name": f"{team} Spinner", "recent_wickets_avg": 1.1, "economy": 7.1},
            ]
        }

    def get_pitch_report(self, venue: str) -> Dict[str, Any]:
        """Returns pitch data based on known Indian venues."""
        venue_lower = venue.lower()
        if "chinnaswamy" in venue_lower:
            return {"type": "Batting paradise", "avg_score": 195, "pace_vs_spin": "Pace struggles"}
        elif "chepauk" in venue_lower or "chidambaram" in venue_lower:
            return {"type": "Spin friendly", "avg_score": 160, "pace_vs_spin": "Spin dominates"}
        elif "wankhede" in venue_lower:
            return {"type": "Batting friendly/Bounce", "avg_score": 185, "pace_vs_spin": "Balanced"}
        else:
            return {"type": "Balanced", "avg_score": 170, "pace_vs_spin": "Neutral"}

    def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetches real weather if key exists, else estimates."""
        if self.weather_api_key:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "forecast": data["weather"][0]["description"],
                        "temp_c": data["main"]["temp"],
                        "humidity": data["main"]["humidity"]
                    }
            except Exception as e:
                print(f"Weather API Error: {e}")
        
        # Fallback simulation
        return {
            "forecast": "Clear Sky (Simulated)",
            "temp_c": 28,
            "humidity": 65,
            "note": "Using historical average for this month"
        }

    def build_context(self, match: MatchContext) -> Dict[str, Any]:
        return {
            "player_form": {
                match.team1: self.get_player_form(match.team1),
                match.team2: self.get_player_form(match.team2)
            },
            "pitch_report": self.get_pitch_report(match.venue),
            "weather": self.get_weather(match.city),
        }