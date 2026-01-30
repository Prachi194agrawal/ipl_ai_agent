# # at top of reasoning_agent.py, rag_agent.py, evaluation_agent.py
# from dotenv import load_dotenv
# load_dotenv()


# import json
# import faiss
# import numpy as np
# from typing import List, Dict, Any
# from openai import OpenAI

# class RAGAgent:
#     def __init__(self, index_path="rag_corpus/index.faiss", jsonl_path="rag_corpus/matches.jsonl"):
#         self.client = OpenAI()
#         self.jsonl_path = jsonl_path
#         self.index_path = index_path
#         self.docs = self._load_docs()
#         self.index, self.embeddings = self._build_or_load_index()

#     def _load_docs(self):
#         docs = []
#         with open(self.jsonl_path) as f:
#             for line in f:
#                 docs.append(json.loads(line))
#         return docs

#     def _embed(self, texts: List[str]) -> np.ndarray:
#         resp = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=texts,
#         )
#         vecs = [item.embedding for item in resp.data]
#         return np.array(vecs, dtype="float32")

#     def _build_or_load_index(self):
 
#         texts = [d["text"] for d in self.docs]
#         embeddings = self._embed(texts)
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dim)
#         index.add(embeddings)
#         faiss.write_index(index, self.index_path)
#         return index, embeddings

#     def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
#         q_emb = self._embed([query])
#         _, idxs = self.index.search(q_emb, k)
#         results = []
#         for idx in idxs[0]:
#             results.append(self.docs[int(idx)])
#         return results

#     def answer_with_context(self, query: str) -> str:
#         docs = self.retrieve(query, k=5)
#         context = "\n".join(d["text"] for d in docs)
#         prompt = f"""
# You are an IPL statistician. Use ONLY the context to answer.

# Context:
# {context}

# Question: {query}

# Answer concisely, citing numbers or patterns from the context.
# """
#         resp = self.client.responses.create(
#             model="gpt-4.1-mini",
#             input=prompt,
#             max_output_tokens=300,
#         )
#         return resp.output[0].content[0].text



# import os, json
# import numpy as np
# from typing import List, Dict, Any
# from dotenv import load_dotenv

# load_dotenv()

# class RAGAgent:
#     def __init__(self, index_path="rag_corpus/index.faiss", jsonl_path="rag_corpus/matches.jsonl"):
#         self.jsonl_path = jsonl_path
#         self.index_path = index_path
#         self.docs = self._load_docs()
#         self.index = None
#         self.embeddings = None
#         print(f"Loaded {len(self.docs)} RAG documents (offline mode)")

#     def _load_docs(self):
#         docs = []
#         try:
#             with open(self.jsonl_path) as f:
#                 for line in f:
#                     docs.append(json.loads(line))
#         except FileNotFoundError:
#             # Fallback to hardcoded docs
#             docs = [
#                 {"id": 1, "text": "2019 Final: MI vs CSK at Hyderabad. MI won by 1 run."},
#                 {"id": 2, "text": "2018 Final: CSK vs SRH at Wankhede. CSK won by 8 wickets."},
#                 {"id": 3, "text": "Wankhede: Batting friendly, avg 1st innings 175."},
#                 {"id": 4, "text": "CSK vs MI: MI leads 20-18 in head-to-head."},
#                 {"id": 5, "text": "Chinnaswamy: High scoring, dew favors chasing."},
#             ]
#         return docs

#     def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
#         """Simple keyword matching (no embeddings)"""
#         query_lower = query.lower()
#         scores = []
#         for doc in self.docs:
#             score = sum(1 for word in query_lower.split() if word in doc["text"].lower())
#             scores.append((score, doc))
#         scores.sort(reverse=True)
#         return [doc for _, doc in scores[:k]]

#     def answer_with_context(self, query: str) -> str:
#         docs = self.retrieve(query, k=3)
#         context = "\n".join(d["text"] for d in docs)
#         return (
#             f"Historical context for '{query}':\n\n"
#             f"{context}\n\n"
#             "(RAG embeddings disabled due to quota. ML prediction still active.)"
#         )



import os, json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGAgent:
    def __init__(self, index_path="rag_corpus/index.faiss", jsonl_path="rag_corpus/matches.jsonl"):
        self.jsonl_path = jsonl_path
        self.index_path = index_path
        self.docs = self._load_docs()
        print(f"Loaded {len(self.docs)} RAG documents (offline mode)")

    def _load_docs(self):
        docs = []
        try:
            with open(self.jsonl_path) as f:
                for line in f:
                    docs.append(json.loads(line))
        except FileNotFoundError:
            docs = [
                {"id": 1, "text": "2019 Final: MI vs CSK at Hyderabad. MI won by 1 run."},
                {"id": 2, "text": "2018 Final: CSK vs SRH at Wankhede. CSK won by 8 wickets."},
                {"id": 3, "text": "Wankhede: Batting friendly, avg 1st innings 175."},
                {"id": 4, "text": "CSK vs MI: MI leads 20-18 in head-to-head."},
                {"id": 5, "text": "Chinnaswamy: High scoring, dew favors chasing."},
            ]
        return docs

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword matching (no embeddings)"""
        query_lower = query.lower()
        scores = []
        for doc in self.docs:
            score = sum(1 for word in query_lower.split() if word in doc["text"].lower())
            scores.append((score, doc))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]

    def answer_with_context(self, query: str) -> str:
        docs = self.retrieve(query, k=3)
        context = "\n".join(d["text"] for d in docs)
        return (
            f"Historical context for '{query}':\n\n"
            f"{context}\n\n"
            "(RAG embeddings disabled due to quota. ML prediction active.)"
        )
