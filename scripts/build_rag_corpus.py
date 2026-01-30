import pandas as pd
from tqdm import tqdm
import json, os

matches = pd.read_csv("data/matches.csv")

os.makedirs("rag_corpus", exist_ok=True)
with open("rag_corpus/matches.jsonl", "w") as f:
    for _, row in tqdm(matches.iterrows(), total=len(matches)):
        text = (
            f"Season {row['season']}: {row['team1']} vs {row['team2']} at "
            f"{row['venue']}. Winner: {row['winner']}. Toss: {row['toss_winner']} "
            f"chose {row['toss_decision']}."
        )
        doc = {"id": int(row["id"]), "text": text}
        f.write(json.dumps(doc) + "\n")
