from llama_cpp import Llama

'''
Not recommended to use this as the source file
Instead I recommend to use the python notebook
for consistent serialization and use of the model
'''

# Load the GGUF model from the same directory
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_0.gguf")

docs = [
    "Install Python: Download and install Python from python.org, then add it to PATH.",
    "Set up a virtual environment: Run 'python -m venv env' and activate it using 'source env/bin/activate' (Mac/Linux) or 'env\\Scripts\\activate' (Windows).",
    "Install dependencies: Use 'pip install -r requirements.txt' to install project dependencies.",
    "Debugging in Python: Use 'import pdb; pdb.set_trace()' to add breakpoints in your code.",
    "Git commit: Stage changes with 'git add .' and commit with 'git commit -m \"Your message\"'.",
    "Create a new Git branch: Use 'git checkout -b branch_name' to create and switch to a new branch.",
    "Merge branches in Git: Use 'git merge branch_name' while on the main branch.",
    "Fix merge conflicts: Open conflicting files, edit manually, then run 'git add .' and 'git commit'.",
    "Set up Flask app: Create 'app.py', install Flask with 'pip install flask', and run with 'flask run'.",
    "Connect to a database: Use SQLAlchemy in Python with 'from sqlalchemy import create_engine'.",
    "Dockerize an application: Create a 'Dockerfile', build with 'docker build -t app .' and run with 'docker run -p 5000:5000 app'.",
    "Deploy a web app: Use services like Heroku, AWS, or DigitalOcean to deploy applications.",
    "Write unit tests: Use 'unittest' or 'pytest' for writing and running test cases.",
    "REST API best practices: Use proper HTTP methods (GET, POST, PUT, DELETE) and return meaningful status codes.",
    "Optimize SQL queries: Use indexing, avoid SELECT *, and use EXPLAIN ANALYZE to check performance.",
    "Handle errors in Python: Use 'try-except' blocks to catch exceptions and log errors.",
    "Use environment variables: Store secrets in '.env' and load them with 'dotenv' or 'os.getenv()'.",
    "Create a Python package: Structure your project, add '__init__.py', and use 'setup.py' to package it.",
    "Optimize Python performance: Use list comprehensions, generators, and built-in functions for efficiency.",
    "Write clean code: Follow PEP 8 guidelines, use meaningful variable names, and document functions properly."
]

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load an embedding model (fast & small)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert docs to embeddings
doc_embeddings = embedding_model.encode(docs)

# Create a FAISS index (L2 distance for similarity)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))  # Add document embeddings

def retrieve_relevant_docs(query, top_k=1):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [docs[i] for i in indices[0]]

while True:
    # Example user query
    query = input("Enter query: ")
    print(query)
    retrieved_docs = retrieve_relevant_docs(query)

    # Augment query with retrieved info
    context = "\n".join(retrieved_docs)
    final_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    print(final_prompt)

    response = llm(final_prompt, max_tokens=50)
    print(response["choices"][0]["text"])
