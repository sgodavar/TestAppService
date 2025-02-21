from flask import Flask, jsonify, request
import json
import os
import pandas as pd
#import openai
from openai import OpenAI


from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

from flask_cors import CORS
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, ValidationError
from langchain.docstore.document import Document


client = OpenAI()

app = Flask(__name__)
CORS(app, origins="*")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY found in environment variables")

#OpenAI = OPENAI_API_KEY
llm = OpenAI(model="gpt-4", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm2 = ChatOpenAI(
    model="gpt-4", 
    temperature=0.7, 
    max_tokens=1000,
    openai_api_key=OPENAI_API_KEY 
)
memory = ConversationBufferMemory(return_messages=True)

# Initialize OpenAI Embeddings and FAISS Vector Store
embedding_model = OpenAIEmbeddings()
vector_store = None  # Placeholder for FAISS vector store
metadata_store = []  # To store metadata for each embedding




###############

# Directory to save uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#################

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
                embedding_model, 
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
                embedding_function=embedding_model,
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
        context = "\n".join([f"{item['content']}" for item in response_data[:3]])

        # Construct system and user messages
        messages = [
            SystemMessage(content="You are an assistant helping with queries on incident data."),
            HumanMessage(content=f"Context:\n{context}\n\nQuery: {query}\nProvide a concise and helpful response."),
        ]

        # Generate response using ChatOpenAI
        response = llm2(messages)

        # Extract and return the assistant's reply
        return response.content.strip()

    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"
    
#########################################################################################
#API Endpoints


@app.route('/api/data', methods=['GET'])
def get_data():

    data = {"message": "Hello from App!"}
    return jsonify(data)



@app.route("/process-excel", methods=["POST"])
def process_excel():
    """
    Process Excel file and add embeddings to FAISS index.
    """
    if vector_store is None:
        return jsonify({"error": "FAISS vector store is not initialized."}), 500
    
    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400

    try:
        excel_path = "./uploads/" + user_input_file_name
        if not os.path.exists(excel_path):
            return jsonify({"error": f"Excel file not found at {excel_path}"}), 404

        df = pd.read_excel(excel_path)
        documents = []
        for _, row in df.iterrows():
            text_representation = ", ".join(f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col]))
            documents.append(Document(page_content=text_representation, metadata=row.to_dict()))

        vector_store.add_documents(documents)
        vector_store.save_local("faiss_index")

        return jsonify({"message": f"Successfully processed {len(df)} rows from the Excel file."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/query_ai", methods=["POST"])
def query_index():
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
        results = vector_store.similarity_search_with_score(user_message, k=4)  # Retrieve results with scores

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
    

@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == 'Test':
        return jsonify({'error': 'No file selected'}), 400

    # Save file to the upload folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    return jsonify({'filePath': f'{file_path}'}), 200


@app.route("/get-excel-data", methods=["GET"])
def get_excel_data():
    """
    Returns the first 10 rows from the uploaded Excel file.
    """
    try:
        # Get file path from query parameters
        file_path = request.args.get("file_path")

        if not file_path:
            return jsonify({"error": "File path is required as a query parameter."}), 400
        

        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": f"Excel file not found at {file_path}"}), 404

        # Read the Excel file
        df = pd.read_excel(file_path)

        df = df.where(pd.notnull(df), None)

        # Get the first 10 rows as a list of dictionaries
        data = df.head(10).to_dict(orient="records")

        # Return data in the desired format
        #print(jsonify(data))
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/chat", methods=["POST", "OPTIONS"])

def chat():
    try:
        if request.method == "OPTIONS":  # Handle preflight
            return jsonify({"message": "Preflight check passed"}), 200
        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Generate response
        response = llm2(memory.chat_memory.messages)

        #print(response.content);

        # Add AI response to memory
        memory.chat_memory.add_message(AIMessage(content=response.content))

        return jsonify({"response": response.content})
    except Exception as e:
        return jsonify({"error": e}), 500
    


    


@app.route('/api/analyse-columns', methods=['POST'])
def post_data():
    # Example to handle POST request
    content = request.json
    response = {"received": content}
    completion = client.chat.completions.create(
       model = "gpt-4o",
       store = True,
       temperature = 0.7,
       response_format = { "type": "json_object" },
       messages=[
         {"role": "system", "content": "You are a helpful assistant."},
           {
            "role": "user",
            "content": "Write a haiku about recursion in programming.Return in JSON format."
           } 
        ]
    )    
    print(completion.choices[0].message)
    # Extract the content of the message
    message_content = completion.choices[0].message.content

        # Return the message content as JSON
    return jsonify({"response": message_content})
   

    

if __name__ == '__main__':
    app.run(debug=True)


