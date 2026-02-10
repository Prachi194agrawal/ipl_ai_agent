import json
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
csv_path = repo_root / "data" / "matches.csv"
output_path = repo_root / "rag_corpus" / "matches.jsonl"

df = pd.read_csv(csv_path)

with open(output_path, "w", encoding="utf-8") as f:
    for i, row in df.head(500).iterrows():  # limit for demo; extend later
        text = (
            f"Match: {row['team1']} vs {row['team2']} at {row['venue']}. "
            f"Winner: {row['winner']}. Toss: {row['toss_winner']} chose {row['toss_decision']}."
        )
        obj = {"id": int(i), "text": text}
        f.write(json.dumps(obj) + "\n")
