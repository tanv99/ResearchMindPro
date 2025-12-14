from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Delete old index
index_name = "researchmind-papers"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"Deleted old index: {index_name}")

# Create new with correct dimensions
pc.create_index(
    name=index_name,
    dimension=384,  # Match SentenceTransformer
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

print(f"Created new index with 384 dimensions")