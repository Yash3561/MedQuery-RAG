# 🩺 MedQuery-RAG: An S-Tier AI Engineering Showcase

![MedQuery-RAG Demo GIF](URL_TO_YOUR_DEMO_GIF_HERE)

## 🚀 Mission

MedQuery-RAG is an end-to-end demonstration of a modern, production-grade AI system. It's designed to be a secure, verifiable, and high-performance knowledge engine that answers complex medical questions by reasoning over a private knowledge base, built entirely with open-source tools.

---

## ✨ Key Features & Demonstrated Skills

*   **Advanced Conversational AI:** The system uses an **LLM-based Intent Classifier** to route user queries, enabling natural conversation while reserving the powerful RAG pipeline for medical questions. Conversational memory is maintained via an **LLM-powered Query Rewriter**.
*   **High-Performance Inference:** The backend is powered by **vLLM**, leveraging techniques like PagedAttention to achieve a **5-10x increase in inference throughput** compared to standard Hugging Face pipelines.
*   **Scalable Data Engineering:** The knowledge base was created using a **synthetic data generation pipeline**, where Llama 3 itself was used to author a comprehensive, 200+ entry medical encyclopedia, which is then indexed into a **FAISS** vector store.
*   **Precision Enhancement & Safety:** A **Cross-Encoder re-ranking model** acts as a secondary filter to improve context relevance. A **confidence score threshold** is used as a safety gate to prevent the model from answering low-relevance queries.
*   **Professional UX/UI:** The entire system is wrapped in a responsive and interactive **Streamlit** web application, featuring a ChatGPT-style interface and transparent source-viewing capabilities with confidence scores.

---

## 🛠️ Architecture

`User Query` -> `Intent Classifier` -> `Query Rewriter` -> `FAISS Retriever` -> `Cross-Encoder Re-ranker` -> `Confidence Gate` -> `Llama 3 Generator` -> `Streamlit UI`

---

## 💻 Tech Stack

*   **Core LLM:** `meta-llama/Llama-3.1-8B-Instruct`
*   **Inference Engine:** `vLLM`
*   **Embedding Model:** `BAAI/bge-large-en-v1.5`
*   **Vector Store:** `FAISS`
*   **Re-ranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **Web Framework:** `Streamlit`

---

## ⚙️ Setup & Running the Project

This project is designed to run on a machine with a powerful NVIDIA GPU (e.g., A100, 4090).

**1. Clone & Setup Environment:**
```bash
git clone https://github.com/your-username/Medquery-Project.git
cd Medquery-Project
conda create -n medquery python=3.10 -y
conda activate medquery
pip install -r requirements.txt
```

**2. Set Up Environment Variables:**
You need a Hugging Face token. Add these lines to your `~/.bashrc` file.
```bash
export HUGGING_FACE_HUB_TOKEN="hf_your_token_here"
# Create a cache directory inside the project to store models
mkdir hf_cache
export HF_HOME=$(pwd)/hf_cache
```
Then run `source ~/.bashrc`.

**3. Generate the Knowledge Base (One-time, ~1 hour task):**
This script uses the LLM to create the dataset and builds the FAISS index.
```bash
python generate_synthetic_data.py
python 1_process_data.py
python 2_build_vector_store.py
```

**4. Run the Application:**
The application is designed for a remote server, requiring two terminals.

*   **Terminal 1 (The Server):** Run the Streamlit app in headless mode.
    ```bash
    streamlit run app.py --server.headless true
    ```

*   **Terminal 2 (The Tunnel - on your local machine):** Create an SSH tunnel. Replace `node_name` and `user` as needed.
    ```bash
    ssh -L 8501:node_name:8501 user@your_hpc_login_node
    ```

*   **View the App:** Open a browser on your local machine and go to `http://localhost:8501`.
