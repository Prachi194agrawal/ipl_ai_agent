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















import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGAgent:
    """
    RAG Agent using LangChain with HuggingFace embeddings and FAISS vector store.
    Integrates with Google Gemini for contextual question answering.
    """
    def __init__(self, data_path="rag_corpus/matches.jsonl"):
        self.data_path = data_path
        print("🔧 RAG: Initializing LangChain with HuggingFace embeddings...")
        
        # 1. Initialize HuggingFace Embeddings (LangChain wrapper)
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Load documents
        self.documents = self._load_documents()
        
        # 3. Create FAISS vector store using LangChain
        self.vectorstore = self._create_vectorstore()
        
        # 4. Initialize LLM for QA (optional, for answer_with_context)
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            self.qa_chain = self._create_qa_chain()
        else:
            self.llm = None
            self.qa_chain = None
        
        print(f"✅ RAG: Indexed {len(self.documents)} documents with LangChain")

    def _load_documents(self) -> List[Document]:
        """Load documents from JSONL file and convert to LangChain Document format"""
        docs = []
        
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        docs.append(Document(
                            page_content=data.get("text", ""),
                            metadata={"id": data.get("id", 0)}
                        ))
        else:
            # Default fallback documents
            default_texts = [
                "2023 Final: CSK vs GT. CSK won by 5 wickets (DLS method). Jadeja hit winning runs.",
                "CSK Record at Chepauk: Win rate 70%+. Spinners play a crucial role.",
                "MI vs CSK Head-to-Head: MI leads historically, but CSK has won 3 of last 4.",
                "Virat Kohli against spin: Strike rate drops to 115 in middle overs.",
                "Dew Factor: In night matches at Wankhede, team batting second wins 60% of times.",
                "RCB at Chinnaswamy: High-scoring venue, average first innings 190+.",
                "Mumbai Indians: 5-time IPL champions, strongest team in playoffs.",
                "Dhoni's record chasing: CSK wins 60% when chasing under Dhoni's captaincy."
            ]
            docs = [Document(page_content=text, metadata={"id": i}) 
                   for i, text in enumerate(default_texts)]
        
        return docs

    def _create_vectorstore(self) -> FAISS:
        """Create FAISS vector store using LangChain"""
        if not self.documents:
            # Create dummy document if empty
            self.documents = [Document(page_content="No data available", metadata={"id": 0})]
        
        vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        return vectorstore

    def _create_qa_chain(self):
        """Create LangChain QA chain using LCEL (LangChain Expression Language)"""
        template = """You are an IPL cricket statistics expert. Use ONLY the provided context to answer questions.

Context:
{context}

Question: {question}

Provide a concise answer based strictly on the context above. If the context doesn't contain relevant information, say "No historical data available for this query."

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Define chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create the chain: retriever -> format -> prompt -> llm -> parser
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return qa_chain

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents using LangChain FAISS similarity search"""
        if not self.vectorstore:
            return ["No historical data available."]
        
        try:
            # Use LangChain's similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"⚠️ RAG retrieval error: {e}")
            return ["Error retrieving historical data."]

    def answer_with_context(self, query: str) -> str:
        """Answer query using LangChain QA chain (if LLM available) or simple retrieval"""
        if self.qa_chain:
            try:
                # Use LangChain QA chain with LCEL
                result = self.qa_chain.invoke(query)
                return result
            except Exception as e:
                print(f"⚠️ QA chain error: {e}, falling back to simple retrieval")
        
        # Fallback: Simple retrieval without LLM
        snippets = self.retrieve(query, k=3)
        return "\n".join(f"- {s}" for s in snippets)


# import os, json
# from typing import List, Dict, Any
# from dotenv import load_dotenv

# load_dotenv()

# class RAGAgent:
#     def __init__(self, index_path="rag_corpus/index.faiss", jsonl_path="rag_corpus/matches.jsonl"):
#         self.jsonl_path = jsonl_path
#         self.index_path = index_path
#         self.docs = self._load_docs()
#         print(f"Loaded {len(self.docs)} RAG documents (offline mode)")

#     def _load_docs(self):
#         docs = []
#         try:
#             with open(self.jsonl_path) as f:
#                 for line in f:
#                     docs.append(json.loads(line))
#         except FileNotFoundError:
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
        
#         scores.sort(key=lambda x: x[0], reverse=True)
#         return [doc for _, doc in scores[:k]]

#     def answer_with_context(self, query: str) -> str:
#         docs = self.retrieve(query, k=3)
#         context = "\n".join(d["text"] for d in docs)
#         return (
#             f"Historical context for '{query}':\n\n"
#             f"{context}\n\n"
#             "(RAG embeddings disabled due to quota. ML prediction active.)"
#         )
