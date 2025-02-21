


from .marvinFunctions import * 
import pandas as pd
import json
import time
import os
########################################################################################

def async_construct_combined(parsed_incidents, df, fileName):
    print('combined task started')
    timelines = []
    classifications = []
    mttds = []
    mttks = []
    recommendations = []
    maturity_analysis = []
    i = 1
    for incident in parsed_incidents:
        attempts = 0
        while attempts < 3:
            try:

                combined: Combined = construct_timeline_classification_mttd_mttk(incident['Short description'],
                                                                  incident['Description'],
                                                                  incident['Resolution notes'])      
                print(combined)
                if combined is not None:
                    timeline = combined.action
                    classification = combined.classification
                    mttk = combined.mttk
                    mttd = combined.mttd
                    # Convert each action to a dictionary
                    timeline_dicts = [action_to_dict(action) if isinstance(action, Action) else action for action in timeline]
                    timelines.append(json.dumps(timeline_dicts))
                    classifications.append(classification.value)
                    mttds.append(mttd)
                    mttks.append(mttk)
                else:
                    timelines.append(None)
                break
            except Exception as e:
                attempts += 1
                print(f"Error processing incident {i}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    timelines.append(None)
        print("Parsed incident: ", i)

        # Introduce a 60-second break after every 50 incidents
        if i % 50 == 0:
            print("Taking a 60-second break...")
            time.sleep(1)

        i += 1
            # Add timelines to the dataframe
    df['AI-Combined'] = timelines
    df['Classification'] = classifications
    df['MTTD'] = mttd
    df['MTTK'] = mttk

    file_name_without_extension = os.path.splitext(fileName)[0]
    output_file_path = './uploads/' + fileName

    if any('start' in key.lower() for key in incident.keys()):
            df['MTTD'] = (pd.to_datetime(df['Opened']) - pd.to_datetime(df['Start'])).dt.total_seconds() / 60
    
    
    df.to_excel(output_file_path, index=False)
    print('Combined task completed')



def async_construct_column_description(df):

    data_files = []
    for column in df.columns:
        for value in df[column].dropna().unique():
            if isinstance(value, (datetime, pd.Timestamp)):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            data_files.append(DataFile(columnName=column, value=str(value)))


    columnDescriptions = get_column_descriptions(data_files)


    return columnDescriptions


def async_construct_timeline(parsed_incidents, df, fileName):
    print('Background task started')
    timelines = []
    i = 1
    for incident in parsed_incidents:
        attempts = 0
        while attempts < 3:
            try:
                if 'Timeline' in incident and incident['Timeline']:
                    print('Using timeline')
                    timeline = construct_timeline_with_timeline_notes(incident['Timeline'])
                else:
                    timeline = construct_timeline(incident['Short description'],
                                                  incident['Description'],
                                                  incident['Resolution notes'])
                print(timeline)
                if timeline is not None:
                    # Convert each action to a dictionary
                    timeline_dicts = [action_to_dict(action) if isinstance(action, Action) else action for action in timeline]
                    timelines.append(json.dumps(timeline_dicts))
                else:
                    timelines.append(None)
                break
            except Exception as e:
                attempts += 1
                print(f"Error processing incident {i}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    timelines.append(None)
        print("Parsed incident: ", i)

        # Introduce a 60-second break after every 50 incidents
        if i % 50 == 0:
            print("Taking a 60-second break...")
            time.sleep(1)

        i += 1
            # Add timelines to the dataframe
    df['Timeline-AI'] = timelines

    file_name_without_extension = os.path.splitext(fileName)[0]
    output_file_path = './uploads/' + fileName

    if any('start' in key.lower() for key in incident.keys()):
            df['MTTD'] = (pd.to_datetime(df['Opened']) - pd.to_datetime(df['Start'])).dt.total_seconds() / 60
    
    
    df.to_excel(output_file_path, index=False)
    print('Background task completed')



def async_construct_classification(df, fileName):
    # Initialize an empty list to store classifications
    classifications = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        attempts = 0
        while attempts < 3:
            try:
                classification = classify_incident(row['Short description'], row['Timeline-AI'])
                classifications.append(classification)
                print(f"Processed row {index}")
                break
            except Exception as e:
                attempts += 1
                print(f"Error classifying incident at index {index}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    classifications.append(None)
        if (index + 1) % 50 == 0:
            time.sleep(5)

    print(classifications)

    # Add the classifications to the dataframe
    df['Classification'] = [c.classification if c else None for c in classifications]
    df['Reason'] = [c.reason if c else None for c in classifications]


    file_name_without_extension = os.path.splitext(fileName)[0]
    output_file_path = './uploads/' + fileName

    df.to_excel(output_file_path, index=False)



def async_top_recommendations(df, fileName):

    recommendations = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        attempts = 0
        while attempts < 3:
            try:
                recommendation = top_recommendations(row['Timeline-AI'], row['Classification'], row['Reason'])
                recommendations.append(recommendation)
                print(f"Processed row {index}")
                break
            except Exception as e:
                attempts += 1
                print(f"Error requesting recommendation for incident at index {index}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    recommendations.append(None)
        if (index + 1) % 50 == 0:
            time.sleep(5)


    #recommendations = [[Recommendation(idea='Review ELB connection error logs.', reason='Understanding the source of errors will help diagnose the issue.'), Recommendation(idea='Check recent changes in backend configuration.', reason='Recent changes could have introduced errors affecting the ELB.'), Recommendation(idea='Analyze network traffic patterns during the error period.', reason='Identifying unusual traffic patterns can indicate the root cause of the issue.'), Recommendation(idea='Monitor ELB performance metrics regularly.', reason='Continuous monitoring can help prevent future occurrences.'), Recommendation(idea='Enhance alerting mechanisms for error thresholds.', reason='Improved alerting can lead to faster issue identification and resolution.')], [Recommendation(idea='Investigate and fix the list export functionality issue.', reason='The error encountered seems to be directly related to the functionality of exporting a list, which indicates an issue with the software system itself.'), Recommendation(idea='Perform a detailed analysis of the system logs during the time of the reported issue.', reason="This will help identify any anomalies or errors occurring during the user's export attempt."), Recommendation(idea='Develop and deploy a patch to address any identified bugs.', reason='This will resolve the root cause of the issue and prevent future occurrences.'), Recommendation(idea='Update the customer once the issue is resolved and provide any necessary guidance or documentation.', reason='Ensuring customer satisfaction and keeping them informed of the progress and resolution.'), Recommendation(idea='Implement automated monitoring for the export functionality to quickly detect and address similar issues in the future.', reason='Proactive monitoring can help in early detection and resolution of system-related issues, improving system reliability.')]]

    # Flatten the 2D recommendations array
    #flattened_recommendations = [item for sublist in recommendations for item in sublist]

    # Add the flattened recommendations to the dataframe
    df['recommendation'] = recommendations
    
    print(recommendations)

     # write the updated dataframe to a new Excel file
    output_file_path = './uploads/' + fileName

    df.to_excel(output_file_path, index=False)

    return recommendations


def async_merge_top_recommendations(recommendationList):

    top_recommendations = []

    top_recommendations = top_recommendations_summary(recommendationList)

    topRecommendationsMerged = [{"idea": item.idea, "reason": item.reason, "business_impact":item.business_impact} for item in top_recommendations]

    print(topRecommendationsMerged)

    return topRecommendationsMerged

def async_get_maturity_analysis(df):
    
    maturity_analysis = []

    maturity_analysis = get_maturity_analysis(df['Timeline-AI'], df['Classification'])

    #MaturityAnalysis = namedtuple('MaturityAnalysis', ['category', 'maturityLevel'])

    react_component_input = {item.category: f"{item.maturityLevel}/{item.reason}" for item in maturity_analysis}

    # Output the result
    #print(react_component_input)

    return react_component_input

def async_get_mttk_from_timeline(df, filePath):
    
    mttks = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        attempts = 0
        while attempts < 3:
            try:
                mttk = time_to_know(row['Timeline-AI'])
                mttks.append(mttk.time_to_know)
                print(f"Processed row {index}")
                break
            except Exception as e:
                attempts += 1
                print(f"Error finding mttk incident at index {index}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    mttks.append(None)
        if (index + 1) % 50 == 0:
            time.sleep(1)

    print(mttks)

    df['MTTK'] = mttks
    # df['MTTK'] = [t.time_to_know if t else None for t in mttks]
    # df['First_Detected'] = [t.first_detected if t else None for t in mttks]
    # df['Cause_Identified'] = [t.cause_identified if t else None for t in mttks]

    # # Ensure columns are in datetime format
    # df['MTTK'] = pd.to_datetime(df['MTTK'], errors='coerce')
    # df['First_Detected'] = pd.to_datetime(df['First_Detected'], errors='coerce')
    # df['Cause_Identified'] = pd.to_datetime(df['Cause_Identified'], errors='coerce')

    # # Remove timezone information if present
    # df['MTTK'] = df['MTTK'].dt.tz_localize(None)
    # df['First_Detected'] = df['First_Detected'].dt.tz_localize(None)
    # df['Cause_Identified'] = df['Cause_Identified'].dt.tz_localize(None)

     # write the updated dataframe to a new Excel file
    df.to_excel(filePath, index=False)



def get_chart_data(df):
    # Example data from the image (simulating Excel input)
    #data = {"Classification": ["ClassificationType.system", 
                           # "ClassificationType.control", "ClassificationType.environment"]}

    # Create a DataFrame
    df = pd.DataFrame(df["Classification"])

    # Extract the classification value (e.g., 'control', 'system', etc.)
    df['Classification'] = df['Classification'].str.split('.').str[-1]

    # Filter valid classification types based on the enum
    valid_classifications = {"control", "system", "environment"}  # Valid values based on your enum
    df = df[df['Classification'].isin(valid_classifications)]

    # Count occurrences of each classification type
    classification_counts = df['Classification'].value_counts()

    # Convert to the React app format
    output = [
        {"id": idx + 1, "value": count, "label": classification}
        for idx, (classification, count) in enumerate(classification_counts.items())
    ]

    # Output
    return output

def get_mttk_and_priority(df, priorityColumnName):

    grouped_data = df.groupby(priorityColumnName)["MTTK"].mean().reset_index()
    result = grouped_data.to_dict(orient="records")
    print(result)

    return result



def chat_with_ai(df, question):
    if "plot" in question.lower() or "plotting" in question.lower() or "compare" in question.lower():
        result = [{"key": key, "value": str(value)} for key, value in df['Summary-From-AI'].items()]
        print("In plot function")
        chat_response = chat(question, df['Summary-From-AI'])
    else:
        chat_response = chat_for_text_response(question, df['Summary-From-AI'])
    
    return chat_response




def generate_summary_for_each_row(df, fileName):
    print('Background task started')
    Summaries = []
    i = 1
    for pos, incident in df.iterrows():
        attempts = 0
        while attempts < 3:
            try:
                
                print('Using generate Summary')
                summary = generate_summary_column(incident)
                summary_form_ai = generate_summary(summary)
                Summaries.append(summary_form_ai)
                print(summary_form_ai)
                break
            except Exception as e:
                attempts += 1
                print(f"Error processing incident {i}, attempt {attempts}: {e}")
                if attempts < 3:
                    time.sleep(0.5)
                else:
                    Summaries.append(None)
        print("Parsed incident: ", i)

        # Introduce a 60-second break after every 50 incidents
        if i % 50 == 0:
            print("Taking a 60-second break...")
            time.sleep(1)

        i += 1
            # Add timelines to the dataframe
    df['Summary-From-AI'] = Summaries

    output_file_path = './uploads/' + fileName
    
    df.to_excel(output_file_path, index=False)
    print('Background task completed')






#################################################################################

# Helper function to convert Action objects to dictionaries
def action_to_dict(action):
    return {
        'timestamp': action.timestamp.isoformat() if isinstance(action.timestamp, datetime) else action.timestamp,
        'description': action.description
    }

def identify_null_columns(df):
    # Identify columns where all values are null, NaN, or empty
    return df.columns[df.isnull().all() | (df == '').all()]

def convert_to_datetime(unix_timestamp):
    # Convert milliseconds to seconds
    unix_timestamp_seconds = unix_timestamp / 1000
    dt = datetime.fromtimestamp(unix_timestamp_seconds)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# Helper function to convert Action objects to dictionaries
def action_to_dict(action):
    return {
        'timestamp': action.timestamp.isoformat() if isinstance(action.timestamp, datetime) else action.timestamp,
        'description': action.description
    }

def generate_dataset_for_barchart(df, column_name):
    # Group by the column and calculate the mean of MTTD
    dataset = (
        df.groupby(column_name)['MTTD']
        .mean()
        .reset_index()
        .rename(columns={column_name: "label", "MTTD": "value"})
    )
    # Format into the required structure
    return {
        "xAxis": dataset["label"].tolist(),
        "yAxis": dataset["value"].tolist(),
    }



def generate_summary_column(row):
    return row.to_dict()
