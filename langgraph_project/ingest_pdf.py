from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Load PDF
loader = PyPDFLoader("example.pdf")  # Make sure this file exists
documents = loader.load_and_split()

# Create embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to running Qdrant Docker instance
client = QdrantClient(host="localhost", port=6333)

# Create vector store with client
qdrant = Qdrant.from_documents(
    documents=documents,
    embedding=embedding_model,
    url="http://localhost:6333",  # must point to Docker Qdrant
    collection_name="rag_docs"


)

print("✅ Documents successfully ingested into Qdrant (Docker)")


