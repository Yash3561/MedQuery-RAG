# 1_process_data.py (Adapted for our OWN synthetic data)
import pandas as pd
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_FILE_PATH = './data/synthetic_medical_notes.csv'

print(f"Loading our synthetic data from {DATA_FILE_PATH}...")
df = pd.read_csv(DATA_FILE_PATH)
df.dropna(subset=['text'], inplace=True)

print(f"Successfully loaded {len(df)} synthetic medical notes.")

documents = []
for index, row in df.iterrows():
    # Metadata now refers to the condition, which is great for citations!
    metadata = {"source_condition": row['condition']}
    documents.append({"text": row['text'], "metadata": metadata})

print(f"Created {len(documents)} document objects.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
all_chunks = []
for doc in documents:
    chunks = text_splitter.split_text(doc['text'])
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": doc['metadata']})

print(f"Created {len(all_chunks)} text chunks from synthetic data.")

with open('processed_chunks.pkl', 'wb') as f:
    pickle.dump(all_chunks, f)

print("SUCCESS: Processed synthetic chunks saved to processed_chunks.pkl")