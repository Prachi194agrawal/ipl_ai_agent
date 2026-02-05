import streamlit as st
import joblib

# import pandas as pd
# from agents.data_fetch_agent import DataFetchAgent, MatchContext
# from agents.reasoning_agent import ReasoningAgent
# from agents.rag_agent import RAGAgent
# from agents.evaluation_agent import EvaluationAgent

# # Load model and encoders
# model = joblib.load("models/ipl_xgb_model.pkl")
# team_encoder = joblib.load("artifacts/team_encoder.pkl")
# venue_encoder = joblib.load("artifacts/venue_encoder.pkl")

# data_agent = DataFetchAgent()
# reason_agent = ReasoningAgent()
# rag_agent = RAGAgent()
# eval_agent = EvaluationAgent()

# st.title("IPL Insight Agent – Match Outcome & Expert Analysis")

# team_list = sorted(team_encoder.classes_.tolist())
# venue_list = sorted(venue_encoder.classes_.tolist())

# col1, col2 = st.columns(2)
# with col1:
#     team1 = st.selectbox("Team 1", team_list)
# with col2:
#     team2 = st.selectbox("Team 2", team_list)

# venue = st.selectbox("Venue", venue_list)
# toss_winner = st.selectbox("Toss Winner", [team1, team2])
# toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

# city = st.text_input("City", "Mumbai")
# date = st.date_input("Match Date")
# user_question = st.text_input(
#     "Ask anything about this match (e.g., 'Why is Team 1 favoured?')"
# )

# if st.button("Predict & Analyse"):
#     # Encode features
#     t1_enc = team_encoder.transform([team1])[0]
#     t2_enc = team_encoder.transform([team2])[0]
#     toss_enc = team_encoder.transform([toss_winner])[0]
#     venue_enc = venue_encoder.transform([venue])[0]
#     toss_bat = 1 if toss_decision == "bat" else 0

#     X_input = pd.DataFrame([{
#         "team1": t1_enc,
#         "team2": t2_enc,
#         "toss_winner": toss_enc,
#         "toss_bat": toss_bat,
#         "venue": venue_enc
#     }])

#     proba = model.predict_proba(X_input)[:, 1][0]

#     st.subheader("Model Prediction")
#     st.write(f"Win probability for **{team1}**: {proba:.2%}")

#     match_info = {
#         "team1": team1,
#         "team2": team2,
#         "venue": venue,
#         "date": str(date)
#     }

#     # Data fetching context
#     ctx = data_agent.build_context(
#         MatchContext(team1=team1, team2=team2, venue=venue, city=city, date=str(date))
#     )

#     # RAG answer
#     query = user_question or f"How have {team1} and {team2} historically performed at {venue}?"
#     rag_answer = rag_agent.answer_with_context(query)

#     st.subheader("RAG Historical Insight")
#     st.write(rag_answer)

#     # Reasoning
#     reasoning = reason_agent.explain_prediction(match_info, proba, ctx)
#     st.subheader("Expert Reasoning")
#     st.write(reasoning)

#     # Evaluation
#     eval_text = eval_agent.evaluate(
#         match_info=match_info,
#         ml_proba=proba,
#         rag_snippets=rag_answer,
#         reasoning_text=reasoning
#     )
#     st.subheader("Evaluation Agent Feedback")
#     st.write(eval_text)



import pandas as pd

import os
from agents.data_fetch_agent import DataFetchAgent, MatchContext
from agents.reasoning_agent import ReasoningAgent
from agents.rag_agent import RAGAgent
from agents.evaluation_agent import EvaluationAgent

# --- 1. Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    # Load Agents
    agents = {
        "data": DataFetchAgent(),
        "reasoning": ReasoningAgent(),
        "rag": RAGAgent(),
        "eval": EvaluationAgent()
    }
    
    # Load ML Model & Encoders
    try:
        model = joblib.load("models/ipl_xgb_model.pkl")
        team_enc = joblib.load("artifacts/team_encoder.pkl")
        venue_enc = joblib.load("artifacts/venue_encoder.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Run 'python scripts/train_model.py' first.")
        return agents, None, None, None

    return agents, model, team_enc, venue_enc

agents, model, team_encoder, venue_encoder = load_resources()

# --- 2. App UI ---
st.title("🏏 IPL Insight Agent (Real ML + AI)")

st.sidebar.header("Match Setup")
if team_encoder:
    team_list = sorted(team_encoder.classes_.tolist())
    venue_list = sorted(venue_encoder.classes_.tolist())
else:
    team_list = []
    venue_list = []

t1_name = st.sidebar.selectbox("Team 1", team_list, index=0)
t2_name = st.sidebar.selectbox("Team 2", team_list, index=1)
venue_name = st.sidebar.selectbox("Venue", venue_list)
city = st.sidebar.text_input("City", "Mumbai")
date = st.sidebar.date_input("Match Date")

st.sidebar.markdown("---")
toss_winner_name = st.sidebar.selectbox("Toss Winner", [t1_name, t2_name])
toss_decision = st.sidebar.selectbox("Toss Decision", ["Bat", "Field"])

user_question = st.text_input("💬 Ask the Expert:", placeholder="How does the pitch report affect this match?")

if st.button("🚀 Predict & Analyze"):
    if not model:
        st.error("Model not loaded.")
        st.stop()
        
    with st.spinner("Running ML Model & AI Agents..."):
        # A. Prepare Data for Model
        t1_val = team_encoder.transform([t1_name])[0]
        t2_val = team_encoder.transform([t2_name])[0]
        venue_val = venue_encoder.transform([venue_name])[0]
        toss_winner_val = team_encoder.transform([toss_winner_name])[0]
        toss_bat_val = 1 if toss_decision == "Bat" else 0
        
        input_data = pd.DataFrame([[t1_val, t2_val, toss_winner_val, toss_bat_val, venue_val]], 
                                  columns=['team1', 'team2', 'toss_winner', 'toss_bat', 'venue'])
        
        # B. Get Prediction
        proba = model.predict_proba(input_data)[:, 1][0] # Probability of Team 1 winning
        
        # C. Display ML Result
        st.subheader("📊 Model Prediction")
        c1, c2 = st.columns(2)
        c1.metric(f"{t1_name} Win %", f"{proba:.1%}")
        c2.metric(f"{t2_name} Win %", f"{(1-proba):.1%}")
        st.progress(float(proba))

        # D. Get Agent Insights
        match_ctx = MatchContext(team1=t1_name, team2=t2_name, venue=venue_name, city=city, date=str(date))
        context_data = agents["data"].build_context(match_ctx)
        
        match_info = {
            "team1": t1_name, "team2": t2_name, "venue": venue_name,
            "toss_winner": toss_winner_name, "toss_decision": toss_decision
        }

        # E. RAG & Reasoning
        rag_resp = agents["rag"].answer_with_context(user_question or f"{t1_name} vs {t2_name} stats")
        reasoning = agents["reasoning"].explain_prediction(match_info, proba, context_data)
        eval_resp = agents["eval"].evaluate(match_info, proba, rag_resp, reasoning)

        with st.expander("📚 Historical Context (RAG)", expanded=True):
            st.info(rag_resp)
        
        st.subheader("🧠 AI Analysis")
        st.markdown(reasoning)
        
        with st.expander("🛡️ System Evaluation"):
            st.markdown(eval_resp)