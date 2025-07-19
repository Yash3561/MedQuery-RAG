# 2_build_vector_store.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading processed chunks...")
with open('processed_chunks.pkl', 'rb') as f:
    chunks_with_meta = pickle.load(f)

# Separate text for embedding and keep metadata aligned
texts = [item['text'] for item in chunks_with_meta]

print("Loading embedding model from local cache...")
# Point to your local, cached model directory
embedding_model = SentenceTransformer('./models/bge-large-en-v1.5', device='cuda')

print("Generating embeddings... (This will take a while)")
embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
embeddings = embeddings.cpu().numpy().astype('float32')

print(f"Embeddings generated with shape: {embeddings.shape}")

# Build the FAISS index
d = embeddings.shape[1]  # Dimension of vectors
index = faiss.IndexFlatL2(d)
index.add(embeddings)

print(f"FAISS index built. Total vectors in index: {index.ntotal}")

# Save the index and the chunks data separately
faiss.write_index(index, "faiss_index.bin")

# We still need the original text for retrieval
with open('chunks_for_retrieval.pkl', 'wb') as f:
    pickle.dump(chunks_with_meta, f)

print("FAISS index and chunks saved.")