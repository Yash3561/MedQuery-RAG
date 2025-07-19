# engine.py (v5.1 - The Final "Showcase" Edition)
import faiss
import pickle
import numpy as np
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import streamlit as st
import json

# --- Constants ---
RERANKER_SCORE_THRESHOLD = -2.0 # The confidence score threshold for our safety gate

class MedQueryEngine:
    def __init__(self):
        """Initializes the entire MedQueryEngine."""
        print("Engine Initializing...")
        self.llm = None
        self.embedding_model = None
        self.reranker = None
        self.index = None
        self.chunks_with_meta = None
        self.sampling_params = None
        self._load_all_components()
        print("Engine Ready.")

    def _load_all_components(self):
        """Loads all data and models into memory."""
        # Load data assets
        self.index = faiss.read_index("faiss_index.bin")
        with open('chunks_for_retrieval.pkl', 'rb') as f:
            self.chunks_with_meta = pickle.load(f)
        
        # Load all models onto the GPU
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
        self.llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.80 # Use 80% of GPU VRAM
        )

        # Default sampling parameters for the main generation task
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            stop_token_ids=[128001, 128009]
        )

    def _classify_intent(self, query: str) -> str:
        """
        Uses the LLM as a zero-shot classifier to determine user intent.
        This is the "AI Router" of the application.
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert intent classifier. Your job is to analyze the user's query and classify it into one of the following categories: "GREETING", "META_QUESTION", "MEDICAL_QUERY", or "OFF_TOPIC".
You must respond with a JSON object containing a single key "intent" and the category as its value.

**Examples:**
- Query: "Hi there!" -> {{"intent": "GREETING"}}
- Query: "Good morning" -> {{"intent": "GREETING"}}
- Query: "What can you do?" -> {{"intent": "META_QUESTION"}}
- Query: "how does this work" -> {{"intent": "META_QUESTION"}}
- Query: "What are the symptoms of malaria?" -> {{"intent": "MEDICAL_QUERY"}}
- Query: "tell me something about the flu" -> {{"intent": "MEDICAL_QUERY"}}
- Query: "What is the capital of Mongolia?" -> {{"intent": "OFF_TOPIC"}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: "{query}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        intent_params = SamplingParams(temperature=0.0, max_tokens=50)
        outputs = self.llm.generate([prompt], intent_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        try:
            intent_json = json.loads(response_text)
            intent = intent_json.get("intent", "MEDICAL_QUERY").upper()
        except (json.JSONDecodeError,AttributeError):
            print(f"  > WARNING: Failed to parse JSON from intent classifier. Defaulting to MEDICAL_QUERY. Raw: '{response_text}'")
            if "medical" in query.lower() or "symptom" in query.lower() or "treat" in query.lower():
                 return "MEDICAL_QUERY"
            return "OFF_TOPIC"
            
        print(f"Query: '{query}' | Classified Intent: {intent}")
        return intent


    def _rewrite_query_with_history(self, query: str, chat_history: list) -> str:
        """
        Uses the LLM to transform a conversational query into a self-contained,
        factual query for the retrieval system.
        """
        if not chat_history:
            return query

        history_str = "".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n" for msg in chat_history])
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query rewriting expert. Your task is to take a chat history and a new, potentially ambiguous question, and rewrite it into a single, standalone question.

Example:
History:
User: What are the symptoms of malaria?
Assistant: The symptoms include fever, chills, etc.
New Question: and what are the medications for it?
Standalone Question: What are the medications for malaria?
<|eot_id|><|start_header_id|>user<|end_header_id|>
Chat History:
{history_str}
New Question: {query}
Standalone Question:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        rewrite_params = SamplingParams(temperature=0.0, max_tokens=100)
        outputs = self.llm.generate([prompt], rewrite_params)
        rewritten_query = outputs[0].outputs[0].text.strip()
        print(f"Original Query: '{query}' | Rewritten Query: '{rewritten_query}'")
        return rewritten_query

    def answer_query(self, query: str, chat_history: list):
        """The main, end-to-end function that orchestrates the entire RAG pipeline."""
        
        # --- Stage 1: INTENT CLASSIFICATION ---
        intent = self._classify_intent(query)
        
        # Stage 2: ROUTING based on intent
        if intent == "GREETING":
            answer = "Hello! I'm MedQuery, your AI medical knowledge engine. How can I help you with a medical question today?"
            return answer, [], [], 0.0

        elif intent == "META_QUESTION":
            answer = """I am **MedQuery-RAG**, an AI Medical Knowledge Engine. Here's what I can do:
- **Answer Medical Questions:** You can ask me about symptoms, diagnosis, and treatments.
- **Provide Verifiable Sources:** I show the exact text from my knowledge base for every answer.
- **Hold a Conversation:** I can understand follow-up questions.
How can I help you with a medical question?"""
            return answer, [], [], 0.0

        elif intent == "OFF_TOPIC":
            answer = "My apologies, but my knowledge is specialized in medical topics. I can't answer questions about that subject. Please ask me a medical question."
            return answer, [], [], 0.0

        elif intent == "MEDICAL_QUERY":
            # IF IT'S A MEDICAL QUERY, PROCEED WITH THE FULL RAG PIPELINE
            with st.status("Processing your medical query...", expanded=True) as status:
                status.write("🧠 Analyzing conversational context...")
                standalone_query = self._rewrite_query_with_history(query, chat_history)
                
                status.write("📚 Searching knowledge base...")
                query_embedding = self.embedding_model.encode(standalone_query)
                distances, indices = self.index.search(np.array([query_embedding]), 20)
                retrieved_candidates = [self.chunks_with_meta[i] for i in indices[0]]
                
                status.write("🎯 Re-ranking documents for precision...")
                reranker_input_pairs = [[standalone_query, doc['text']] for doc in retrieved_candidates]
                scores = self.reranker.predict(reranker_input_pairs)
                scored_candidates = sorted(zip(scores, retrieved_candidates), key=lambda x: x[0], reverse=True)
                
                highest_score = scored_candidates[0][0] if scored_candidates else -10.0
                print(f"Highest relevance score: {highest_score:.2f}")

                # The Confidence Gate
                if highest_score < RERANKER_SCORE_THRESHOLD:
                    status.update(label="⚠️ No relevant info found", state="complete", expanded=False)
                    return "I couldn't find any specific information about that in my knowledge base. Could you try rephrasing your question?", [], [], highest_score

                final_context_docs = [doc for score, doc in scored_candidates[:5]]
                
                status.write("✍️ Generating final answer...")
                prompt, sources = self.format_prompt(query, chat_history, final_context_docs)
                outputs = self.llm.generate([prompt], self.sampling_params)
                answer = outputs[0].outputs[0].text.strip()
                status.update(label="✅ Query processed!", state="complete", expanded=False)

            return answer, sources, final_context_docs, highest_score
        
        else: # A final fallback
            answer = "I'm having a bit of trouble understanding your request. Could you please rephrase?"
            return answer, [], [], 0.0

    def format_prompt(self, query: str, chat_history: list, retrieved_chunks: list):
        """Builds the final, massive prompt for the main generation step."""
        history_str = "".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n" for msg in chat_history])
        context_str = "".join([f"--- Source {i+1} ---\n{chunk['text']}\n\n" for i, chunk in enumerate(retrieved_chunks)])
        sources = list(set([chunk['metadata']['source_condition'] for chunk in retrieved_chunks]))

        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are MedQuery, an expert AI medical encyclopedia. Your sole purpose is to provide general, factual information about medical conditions based on the context provided.

**Your Core Directives:**
1.  **Answer ONLY from Context:** Base your entire answer on the "Relevant Context from Knowledge Base" provided below. If the context is empty or irrelevant, state that you cannot find the information in your knowledge base.
2.  **Do Not Diagnose or Advise:** You are an encyclopedia, not a doctor. You MUST NOT provide medical advice, diagnoses, or treatment plans for the user. If a question implies a personal medical situation (e.g., "I have a fever," "how should I treat..."), you MUST start your response by stating you cannot provide medical advice and recommend consulting a healthcare professional. You may then provide general information relevant to the topic, if available in the context.

**Your Response Format:**
- Use clear, professional language.
- Use bullet points and bold text to structure information.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Conversation History:
{history_str}
Current Question: {query}

Relevant Context from Knowledge Base:
{context_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt_template, sources