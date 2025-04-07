import pytest
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant  # ✅ NEW correct import!
from langchain_community.embeddings import HuggingFaceEmbeddings
from main import graph

@pytest.fixture(scope="module")
def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)
    return Qdrant(client=client, collection_name="pdf_docs", embeddings=embeddings)

def test_weather_query():
    state = {"input": "What's the weather like in London?"}
    result = graph.invoke(state)["response"]
    assert "London" in result or "weather" in result.lower()

def test_pdf_query(setup_vectorstore):
    state = {"input": "What does the document say about climate change?"}
    result = graph.invoke(state)["response"]
    assert isinstance(result, str)
    assert len(result) > 0

def test_invalid_query():
    state = {"input": "Blah blah unrecognized input"}
    result = graph.invoke(state)["response"]
    assert isinstance(result, str)
    assert len(result) > 0




