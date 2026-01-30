import requests
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class MatchContext:
    team1: str
    team2: str
    venue: str
    city: str
    date: str  # ISO string

class DataFetchAgent:
    def __init__(self):
        pass

    def get_player_form(self, team: str) -> Dict[str, Any]:
        # TODO: integrate proper cricket API; for now use placeholder structure
        # e.g., read from a local CSV of recent scores
        return {
            "team": team,
            "key_players": [
                {"name": "Player A", "recent_runs": 45.3, "recent_wickets": 1.2},
                {"name": "Player B", "recent_runs": 36.8, "recent_wickets": 0.4},
            ]
        }

    def get_pitch_report(self, venue: str) -> Dict[str, Any]:
        # For production, scrape from Cricbuzz/ESPN if allowed.
        return {
            "venue": venue,
            "pitch_type": "batting-friendly",
            "avg_first_innings_score": 175,
            "spin_assistance": "moderate"
        }

    def get_team_composition(self, team: str) -> Dict[str, Any]:
        # Read from local JSON of squads for the season.
        return {
            "team": team,
            "probable_xi": [
                "Batter1", "Batter2", "Batter3",
                "Allrounder1", "Allrounder2",
                "Bowler1", "Bowler2", "Bowler3", "Bowler4"
            ]
        }

    def get_weather(self, city: str, date: str) -> Dict[str, Any]:
        # Integrate with OpenWeatherMap or similar with an API key via env.
        return {
            "city": city,
            "date": date,
            "forecast": "Clear",
            "temperature_c": 30,
            "humidity": 55,
            "rain_probability": 10
        }

    def build_context(self, match: MatchContext) -> Dict[str, Any]:
        return {
            "player_form": {
                match.team1: self.get_player_form(match.team1),
                match.team2: self.get_player_form(match.team2)
            },
            "pitch_report": self.get_pitch_report(match.venue),
            "team_composition": {
                match.team1: self.get_team_composition(match.team1),
                match.team2: self.get_team_composition(match.team2)
            },
            "weather": self.get_weather(match.city, match.date),
        }
