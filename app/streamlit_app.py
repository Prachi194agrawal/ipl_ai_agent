import streamlit as st
import joblib
import pandas as pd
from agents.data_fetch_agent import DataFetchAgent, MatchContext
from agents.reasoning_agent import ReasoningAgent
from agents.rag_agent import RAGAgent
from agents.evaluation_agent import EvaluationAgent

# Load model and encoders
model = joblib.load("models/ipl_xgb_model.pkl")
team_encoder = joblib.load("artifacts/team_encoder.pkl")
venue_encoder = joblib.load("artifacts/venue_encoder.pkl")

data_agent = DataFetchAgent()
reason_agent = ReasoningAgent()
rag_agent = RAGAgent()
eval_agent = EvaluationAgent()

st.title("IPL Insight Agent – Match Outcome & Expert Analysis")

team_list = sorted(team_encoder.classes_.tolist())
venue_list = sorted(venue_encoder.classes_.tolist())

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", team_list)
with col2:
    team2 = st.selectbox("Team 2", team_list)

venue = st.selectbox("Venue", venue_list)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

city = st.text_input("City", "Mumbai")
date = st.date_input("Match Date")
user_question = st.text_input(
    "Ask anything about this match (e.g., 'Why is Team 1 favoured?')"
)

if st.button("Predict & Analyse"):
    # Encode features
    t1_enc = team_encoder.transform([team1])[0]
    t2_enc = team_encoder.transform([team2])[0]
    toss_enc = team_encoder.transform([toss_winner])[0]
    venue_enc = venue_encoder.transform([venue])[0]
    toss_bat = 1 if toss_decision == "bat" else 0

    X_input = pd.DataFrame([{
        "team1": t1_enc,
        "team2": t2_enc,
        "toss_winner": toss_enc,
        "toss_bat": toss_bat,
        "venue": venue_enc
    }])

    proba = model.predict_proba(X_input)[:, 1][0]

    st.subheader("Model Prediction")
    st.write(f"Win probability for **{team1}**: {proba:.2%}")

    match_info = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "date": str(date)
    }

    # Data fetching context
    ctx = data_agent.build_context(
        MatchContext(team1=team1, team2=team2, venue=venue, city=city, date=str(date))
    )

    # RAG answer
    query = user_question or f"How have {team1} and {team2} historically performed at {venue}?"
    rag_answer = rag_agent.answer_with_context(query)

    st.subheader("RAG Historical Insight")
    st.write(rag_answer)

    # Reasoning
    reasoning = reason_agent.explain_prediction(match_info, proba, ctx)
    st.subheader("Expert Reasoning")
    st.write(reasoning)

    # Evaluation
    eval_text = eval_agent.evaluate(
        match_info=match_info,
        ml_proba=proba,
        rag_snippets=rag_answer,
        reasoning_text=reasoning
    )
    st.subheader("Evaluation Agent Feedback")
    st.write(eval_text)
