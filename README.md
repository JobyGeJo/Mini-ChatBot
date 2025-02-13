# 🔍 FAISS-Based Semantic Search with Llama and SentenceTransformers

This project demonstrates **semantic search** using:
- **Llama (via Llama.cpp)** for language model processing
- **SentenceTransformers (all-MiniLM-L6-v2)** for sentence embeddings
- **FAISS (Facebook AI Similarity Search)** for fast similarity search

## 🚀 Features
✅ Loads a **Llama-based language model** (`mistral-7b-instruct-v0.1.Q4_0.gguf`)  
✅ Uses **SentenceTransformers** to generate embeddings for text  
✅ Stores and searches embeddings efficiently with **FAISS**  
✅ Finds the most relevant document snippets based on input queries  

---

## 📦 Installation

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```
### 2️⃣ Set Up a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
## 📌 Usage

### 1️⃣ Run the Jupyter Notebook

jupyter notebook main.ipynb

### 2️⃣ What the Notebook Does

 - Loads a Llama model from a local .gguf file.
 - fines a list of technical documentation snippets.
 - Converts the text into embeddings using SentenceTransformers.
 - Indexes the embeddings with FAISS.
 - Allows querying for similar documents using FAISS.

## ⚡ Example Code Snippet
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documentation
docs = [
    "Install Python: Download and install Python from python.org.",
    "Set up a virtual environment: Run 'python -m venv env' and activate it.",
    "Deploy a web app: Use Heroku, AWS, or DigitalOcean."
]

# Convert docs to embeddings
doc_embeddings = embedding_model.encode(docs)

# Create a FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))  # Add embeddings

# Query for similar docs
query_embedding = embedding_model.encode(["How to deploy an application?"])
distances, indices = index.search(query_embedding, k=2)
print("Most similar document indices:", indices)
```
## 🛠 Technologies Used
 - Python 3.x
 - FAISS (Facebook AI Similarity Search)
 - SentenceTransformers (Hugging Face)
 - Llama.cpp (Meta AI)
 - Jupyter Notebook

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 💡 Future Enhancements
 - Implement real-time API using FastAPI for search queries.
 - Integrate web-based UI for document lookup.
 - Add larger language models for better search relevance.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## ✨ Acknowledgments
 - Meta AI for Llama.cpp
 - Hugging Face for SentenceTransformers
 - Facebook AI for FAISS

---