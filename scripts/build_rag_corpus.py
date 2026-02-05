with open("rag_corpus/matches.jsonl", "w") as f:
    for i, row in df.head(500).iterrows():  # limit for demo; extend later
        text = (
            f"Match: {row['team1']} vs {row['team2']} at {row['venue']}. "
            f"Winner: {row['winner']}. Toss: {row['toss_winner']} chose {row['toss_decision']}."
        )
        obj = {"id": int(i), "text": text}
        f.write(json.dumps(obj) + "\n")
