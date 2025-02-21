import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage
import asyncio
import streamlit as st
import json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import faiss
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.docstore.document import Document


app = Flask(__name__)
CORS(app, origins="*")




load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOpenAI(model="gpt-3.5-turbo")
openai_embeddings = OpenAIEmbeddings()

memory = ConversationBufferMemory(return_messages=True)

# Initialize OpenAI Embeddings and FAISS Vector Store

vector_store = None  # Placeholder for FAISS vector store
metadata_store = []  # To store metadata for each embedding





##Initialize or Load FAISS Index
def initialize_faiss():
    """
    Initialize FAISS vector store.
    """
    global vector_store
    embedding_dim = 1536  # Dimensionality of OpenAI embeddings

    try:
        if os.path.exists("faiss_index"):
            # Load existing FAISS index
            vector_store = FAISS.load_local(
                "faiss_index", 
                openai_embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index.")
        else:
            # Create a new FAISS index
            index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
            docstore = InMemoryDocstore({})  # Use InMemoryDocstore for document management
            index_to_docstore_id = {}  # Empty mapping for a new index

            # Initialize the FAISS vector store
            vector_store = FAISS(
                embedding_function=openai_embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
            )
            print("Initialized new FAISS index.")
    except Exception as e:
        print(f"Error initializing FAISS: {e}")
        vector_store = None

print("Initializing FAISS...")
initialize_faiss()



def generate_ai_response(query, response_data):
    """
    Generate an AI response using the OpenAI ChatCompletion API.
    """
    try:

        
        # Prepare context from FAISS results
        context = "\n".join([f"{item['content']}" for item in response_data[:40]])

        # Construct system and user messages
        messages = [
            SystemMessage(content="You are an assistant helping with queries on incident data."),
            HumanMessage(content=f"Context:\n{context}\n\nQuery: {query}\nProvide a concise and helpful response."),
        ]

        # Generate response using ChatOpenAI
        response = llm(messages)

        # Extract and return the assistant's reply
        return response.content.strip()

    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"
    
def generate_vector():
    excel_path = "../uploads/Demo-Incidents.xlsx"
    if not os.path.exists(excel_path):
        return jsonify({"error": f"Excel file not found at {excel_path}"}), 404
    
    #if not os.path.exists(vector_store_path="faiss_index"):
    print("generating index")
    df = pd.read_excel(excel_path)
    documents = []
    for _, row in df.iterrows():
        text_representation = ", ".join(f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col]))
        documents.append(Document(page_content=text_representation, metadata=row.to_dict()))

    vector_store.add_documents(documents)
    vector_store.save_local("faiss_index")
    
generate_vector()

@app.route("/api/test", methods=["POST"])
def Test():
    """
    Query FAISS vector store and provide AI-generated response with a relevance threshold.
    Input: {"message": "user request to AI model"}
    """
    if vector_store is None:
        return jsonify({"error": "FAISS vector store is not initialized."}), 500

    try:
        # Parse input JSON
        data = request.get_json()
        user_message = data.get("message")

        if not user_message:
            return jsonify({"error": "Message is required in the input JSON."}), 400
        

        

        # Perform similarity search on FAISS
        results = vector_store.similarity_search_with_score(user_message, k=50)  # Retrieve results with scores

        # Log the results and scores for debugging
        #print(f"Similarity Search Results: {results}")
        # Apply relevance threshold (e.g., score >= 0.1 for cosine similarity)
        threshold = 0.1
        filtered_results = [
            {
                "content": result[0].page_content,
                "metadata": result[0].metadata,
                "score": float(result[1])  # Convert float32 to float
            }
            for result in results
            if result[1] >= threshold
        ]

        # Fallback: If no results meet the threshold, return the top result regardless of score
        if not filtered_results:
            filtered_results = [
                {
                    "content": result[0].page_content,
                    "metadata": result[0].metadata,
                    "score": float(result[1])  # Convert float32 to float
                }
                for result in results
            ][:2]  # Return the top 2 results
        print(len(filtered_results))

        # Generate AI response using OpenAI or another AI service
        ai_response = generate_ai_response(user_message, filtered_results)

        # Format the final output
        output = {
            "response": ai_response
        }

        return jsonify(output), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)





