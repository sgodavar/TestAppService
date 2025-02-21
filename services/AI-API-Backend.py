from functions.functions import *
from flask import Flask, send_file, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import json
import os
import pandas as pd
from openai import OpenAI
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib
import io
from typing import Union
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import marvin


app = Flask(__name__, static_folder="static", template_folder="templates")
matplotlib.use('agg')

# Load the OpenAI API key from the environment

app = Flask(__name__)
CORS(app, origins="*")

#####################################################

# Directory to save uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

######################################################
#setup for marvin
# BASE_DIR = Path(".")
# ENV_DIR = BASE_DIR
# DOT_ENV = ENV_DIR / ".envazure1"

# print(DOT_ENV)

#load = load_dotenv("./.marvin.env", override=True)
dotenv_path = Path('.env')
load = load_dotenv(dotenv_path, override=True)

print(load)
#####################################################

#Flask API fucntions

#####################################################

# Serve React's static files (JS, CSS, images)
@app.route('/assets/<path:path>')
def serve_static_files(path):
    return send_from_directory(os.path.join(app.static_folder, "assets"), path)

# Serve React's index.html for all other routes
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react_app(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")



@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Delete existing file if it exists
        # if os.path.exists(file_path):
        #     try:
        #         os.remove(file_path)
        #     except PermissionError:
        #         return jsonify({'error': 'File is being used by another process'}), 400
        file.save(file_path)

        return jsonify({'fileName': f'{file.filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route("/api/get-excel-data", methods=["GET"])
def get_excel_data():
    """
    Returns the first 10 rows from the uploaded Excel file.
    """
    try:
        # Get file path from query parameters
        file_path = request.args.get("file_path")

        if not file_path:
            return jsonify({"error": "File path is required as a query parameter."}), 400
        

        # Read the Excel file
        df = pd.read_excel("./uploads/" + file_path)

        df = df.where(pd.notnull(df), None)

        # Get the first 10 rows as a list of dictionaries
        data = df.head(10).to_dict(orient="records")

        # Return data in the desired format
        #print(jsonify(data))
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/get-excel-columns", methods=["GET"])
def get_excel_columns():
    """
    Returns the first 10 rows from the uploaded Excel file.
    """
    try:
        # Get file path from query parameters
        file_path = request.args.get("file_path")

        if not file_path:
            return jsonify({"error": "File path is required as a query parameter."}), 400
        

        # Read the Excel file
        df = pd.read_excel("./uploads/" + file_path)

        df = df.where(pd.notnull(df), None)

        # Get the first 10 rows as a list of dictionaries
        data = df.columns.tolist()

        # Return data in the desired format
        #print(jsonify(data))
        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/api/column-description', methods=['GET'])
def get_description():


    user_input_file_name = request.args.get("file-name")

    file_path = './uploads/' + user_input_file_name

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400

    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400

    try:
        # Load the data
        df = pd.read_excel(file_path)

        incidents_array = df.to_json(orient='records')
        parsed_incidents = json.loads(incidents_array)

        # Assuming parsed_incidents is a list of dictionaries
        for incident in parsed_incidents:
            if 'Opened' in incident:
                incident['Opened'] = convert_to_datetime(incident['Opened'])


        columnDescriptionResponse = async_construct_column_description(df.head(2))

        print(columnDescriptionResponse)


        return jsonify([desc.dict() for desc in columnDescriptionResponse])

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/reconstruct-timeline', methods=['POST'])
def reconstruct_timeline():
    
    user_input_file_name = request.json.get("file-name")

    file_path = './uploads/' + user_input_file_name

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400

    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400

    try:
        # Load the data
        df = pd.read_excel(file_path)

        incidents_array = df.to_json(orient='records')
        parsed_incidents = json.loads(incidents_array)

        # Assuming parsed_incidents is a list of dictionaries
        for incident in parsed_incidents:
            if 'Opened' in incident:
                incident['Opened'] = convert_to_datetime(incident['Opened'])
            if 'Resolved' in incident:
                incident['Resolved'] = convert_to_datetime(incident['Resolved'])
            if 'Closed' in incident:
                incident['Closed'] = convert_to_datetime(incident['Closed'])
            if 'Start' in incident:
                incident['Start'] = convert_to_datetime(incident['Start'])
       

        async_construct_timeline(parsed_incidents, df, user_input_file_name)


        return jsonify({"analysis": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/construct-classification', methods=['POST'])
def construct_classification():

    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_path = './uploads/' + user_input_file_name


    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400

    try:
        # Load the data
        df = pd.read_excel(file_path)

        classification = async_construct_classification(df, user_input_file_name)


        return jsonify({"analysis": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/api/recommendation', methods=['POST'])
def request_recommendations():


    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_name_without_extension = os.path.splitext(user_input_file_name)[0]

    timeline_file_path = './uploads/' + user_input_file_name


    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(timeline_file_path):
        return jsonify({"error": "File does not exist"}), 400
    

    try:
        # Load the data
        df = pd.read_excel(timeline_file_path)

        rec = async_top_recommendations(df, user_input_file_name)

        return jsonify({"analysis": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


@app.route('/api/merge_recommendation', methods=['POST'])
def request_merge_recommendations():
    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_name_without_extension = os.path.splitext(user_input_file_name)[0]

    file_path = './uploads/' + user_input_file_name


    try:
        # Load the data
        df = pd.read_excel(file_path)

        RecommendationData = df['recommendation'].tolist()

        rec = async_merge_top_recommendations(RecommendationData)

        return jsonify(rec)


    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route('/api/request_maturity_analysis', methods=['POST'])
def request_maturity_analysis():
    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_name_without_extension = os.path.splitext(user_input_file_name)[0]

    file_path = './uploads/' + user_input_file_name


    try:
        # Load the data
        df = pd.read_excel(file_path)

        rec = async_get_maturity_analysis(df)

        return jsonify(rec)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


    
@app.route('/api/request_mttk', methods=['POST'])
def request_mttk_from_timeline():
    user_input_file_name = request.json.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_path = './uploads/' + user_input_file_name


    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Recommendation File does not exist"}), 400

    try:
        # Load the data
        df = pd.read_excel(file_path)

        async_get_mttk_from_timeline(df, file_path)


        return jsonify({"analysis": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@app.route('/api/get_mttk_againt_category', methods=['GET'])
def get_mttk_againt_category():

    user_input_file_name = request.args.get("file-name")
    user_input_priority_column_name = request.args.get("column-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_path = './uploads/' + user_input_file_name
    
    try:
         # Load the data
        df = pd.read_excel(file_path)

        result = get_mttk_and_priority(df, user_input_priority_column_name)

        return jsonify(result)


    except Exception as e:
        return jsonify({"error": str(e)}), 400

    


@app.route('/api/get_pie_chart_data', methods=['GET'])
def get_pie_chart_data():

    user_input_file_name = request.args.get("file-name")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400


    file_name_without_extension = os.path.splitext(user_input_file_name)[0]

    file_path = './uploads/' + user_input_file_name
    
    try:
         # Load the data
        df = pd.read_excel(file_path)

        result = get_chart_data(df)

        return jsonify(result)


    except Exception as e:
        return jsonify({"error": str(e)}), 400
    





@app.route('/api/get-bar-chart-data-for-mttd', methods=['GET'])
def get_bar_chart_data():
    user_input_file_name = request.args.get("file-name")
    file_path = './uploads/' + user_input_file_name

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400

    if not marvin.settings.openai.api_key:
        return jsonify({"error": "No API key"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400
    
    try:
         # Load the data
        df = pd.read_excel(file_path)


        result = df.head(1).to_dict(orient="records")

        # Generate datasets for the required columns
        datasets = {
            "Line of Business": generate_dataset_for_barchart(df, "Line of Business"),
            "Category": generate_dataset_for_barchart(df, "Category"),
            "Sub Category": generate_dataset_for_barchart(df, "Sub Category"),
        }

        # Output the datasets
        print(datasets)


        return jsonify(datasets)


    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
@app.route('/api/chat-with-ai', methods=['POST'])
def chat_ai():
    user_input_file_name = request.json.get("file-name")
    print(request.json)
    file_path = './uploads/' + user_input_file_name
    user_chat_request = request.json.get("question")

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400
    
    if not user_chat_request:
        return jsonify({"error": "No question provided"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400
    
    df = pd.read_excel(file_path)
    

    res = chat_with_ai(df, user_chat_request)
    print(res.response)

    if not hasattr(res, 'plotData'):
        return jsonify({"response": res.response})

    return jsonify({"response":res.response, "data": {"xValues":res.plotData.xAxisData, "yValues":res.plotData.yAxisData}})


@app.route('/api/generate_ai_summary', methods=['POST'])
def generate_ai_summary():
    user_input_file_name = request.json.get("file-name")
    file_path = './uploads/' + user_input_file_name

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400 
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400
    
    df = pd.read_excel(file_path)  

    res = generate_summary_for_each_row(df, user_input_file_name)

    return jsonify({"summary-generation": "success"})




@app.route('/api/plot', methods=['POST'])
def plot_graph():
    # Generate a simple plot
    xAxisValues = request.json.get("xAxis")
    yAxisValues = request.json.get("yAxis")
    fig, ax = plt.subplots()
    try:
        y = list(map(float, yAxisValues))
    except ValueError:
        y = list(map(str, yAxisValues))
    x = xAxisValues
    
    ax.plot(x, y, marker='o', linestyle='-')
    ax.set_title("Incident Analysis")

    # Save the plot to a bytes buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close(fig)  # Close the figure to free memory

    return send_file(img_buf, mimetype='image/png')


@app.route('/api/generate_ai_combined', methods=['POST'])
def generate_ai_combined():
    user_input_file_name = request.json.get("file-name")
    file_path = './uploads/' + user_input_file_name

    if not user_input_file_name:
        return jsonify({"error": "No message provided"}), 400 
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400
    
    df = pd.read_excel(file_path)
    incidents_array = df.to_json(orient='records')
    parsed_incidents = json.loads(incidents_array)  

    res = async_construct_combined(parsed_incidents, df, user_input_file_name)

    return jsonify({"summary-generation": "success"})


    

    
######################################


if __name__ == '__main__':
    app.run(debug=True)