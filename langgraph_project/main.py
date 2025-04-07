import os
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.qdrant import Qdrant
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableLambda
import requests

# === LLM Setup ===
llm = Ollama(model="gemma:2b")

# === Embedding + Vector Store ===
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
qdrant = Qdrant(
    collection_name="pdf_collection",
    embeddings=embeddings,
    url="http://localhost:6333",
)

# === Load and Index PDF ===
def load_and_index_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    qdrant.add_documents(docs)

# === RAG Retriever ===
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant.as_retriever()
)

# === Weather API Function ===
def get_weather(location: str) -> str:
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return "Sorry, couldn't fetch weather data."
    data = response.json()
    weather = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    return f"The weather in {location} is {weather} with {temp}°C."

# === Decision Node ===
def decide_action(state: dict):
    question = state["input"].lower()
    if "weather" in question or "temperature" in question:
        return "weather"
    else:
        return "rag"

# === LangGraph Nodes ===
def rag_node(state: dict):
    question = state["input"]
    result = rag_chain.run(question)
    return {"response": result}

def weather_node(state: dict):
    question = state["input"]
    # Assume location is last word for simplicity
    location = question.split("in")[-1].strip().rstrip("?")
    result = get_weather(location)
    return {"response": result}

# === LangGraph Pipeline ===
builder = StateGraph()
builder.add_node("rag", RunnableLambda(rag_node))
builder.add_node("weather", RunnableLambda(weather_node))
builder.set_entry_point("decide")
builder.add_conditional_edges("decide", decide_action, {
    "weather": "weather",
    "rag": "rag"
})
builder.add_edge("weather", END)
builder.add_edge("rag", END)

graph = builder.compile()

# ✅ Exportable pipeline for test
def run_pipeline(user_input: str) -> str:
    state = {"input": user_input}
    result = graph.invoke(state)
    return result.get("response", "No response")

