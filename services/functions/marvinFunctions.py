
import marvin
from classes.marvinModels import * 


###Marvin Functions
@marvin.fn
def generate_summary(rowSummary: dict) -> str:
    """
    Given the dict of an entire row in a excel, Summarize this to JSON. This data can be used to 
    query any data like chart data (xaxis, yaxis), summary of perticular column and to query on the data. 
    """

@marvin.fn
def time_to_know(timeline: list[Action]) -> TimetoKnowReason:
    """
    Time to know is the time that has elased between when the incident was first detected to when a cause for the incident was identified.
    From the given `timeline`, calculate the time to know in minutes.
    Think carefully and step by step.
    """

@marvin.fn
def get_maturity_analysis(timeline: list[list[Action]], classification: list[ClassificationType]) -> list[MaturityAnalysis]:
    """
    Based on the input containing incident timeline, please generate an input object for a maturity grid analysis. we also need the reasoning behind this and any improvements.
    The input should include categories ('Service Monitoring', 'Metrics SLIs / SLOs', 'Incident / Crisis Mgmt.', 'Build & Release Mgmt', 'Capacity Mgmt.', 'Secure/policy as code', 'Problem Mgmt.', 'Cost Optimization', 'Toil Reduction', 'BCDR') mapped to their respective maturity levels ('Discovering', 'Growth-Focused', or 'Industry-Leading').  Think step by step.
    """


@marvin.fn
def get_column_descriptions(columns: list[DataFile]) -> list[DataFileColumnDescriptions]:
    """
    Given the data file with columnName and its value, provide a description of the column. 
    If any column has relevant info mark ContainsIncidentInformation as yes. For Example: Comments column, information column
    if any column had classificaiton info, ContainsClassificationInformation as yes. Classification info corresponds to Priority, business unit etc
    """

@marvin.fn
def top_recommendations_summary(recommendationToSummarize: list[list[Recommendation]]) -> list[Recommendation]:
    """
    Make me a summary of the recommendations and give me top 5 that are most important. Include the ones which has more business impactThink step by step and explain your reasoning
    """


@marvin.fn
def top_recommendations(timeline: list[Action], classification: ClassificationType, reason:str) -> list[Recommendation]:
    """
    Give me top 1 recommendations based on timeline of actions, classification type and their respective reasoning. 
    Do not give genric recommendations like add monitoring, logging, patching etc.
    The business impact should state what metric this would change like MTTD, MTTK, MTTF etc by x% where relevant. Be specific and techinical. 
    
    Think step by step and explain your reasoning
    """


@marvin.fn
def construct_timeline(short: str, description: str, notes:str) -> list[Action]:
    """
    Construct a timeline of actions from the below incident's short description: `short`, full description: `description`, and resolution notes: `notes`.
    We want to use this timeline to understand the sequence of events that occurred during the incident.
    We want to then help us learn and troubleshoot similar incidents in the future.
    """
def construct_timeline_with_timeline_notes(notes: str) -> list[Action]:
    """
    Construct a timeline of actions from the below incident's Timeline : `notes`. The content might be in HTML. Please parse it accordingly.
    We want to use this timeline to understand the sequence of events that occurred during the incident.
    We want to then help us learn and troubleshoot similar incidents in the future.
    """



@marvin.fn
def classify_incident(description: str, action_list: list[Action]) -> Classification:
    """
    You are a site reliability engineer and an expert at root cause analysis.

    You perform root cause analysis using the following technique.

    You analyze any given issue across 3 separate areas.

    System: The software system that needs to run reliably. This includes your source code, your build release pipelines, your test scripts etc.

    Environment: The environment in which the system operates. This could include network conditions, user behavior, macroeconomic situation, general trends etc.

    Control: This is at the boundary of system and its environment. This could include infrastructure management, DNS, load balancing, capacity planning etc.

    System Vulnerability: These are weaknesses or flaws in a software system that could potentially be exploited.

    Worst Case Input: This refers to specific data or commands sent to the system that specifically trigger the vulnerability.

    System Failure: The result of the vulnerability being exploited.

    Root Cause: The process of identifying the fundamental reason for the system failure.

    Environment Interaction: This emphasizes that the vulnerability is often exposed through the system's interaction with its external environment.


    For the given `description` and `action_list`, classify the incident into one of System, Control, Environment.

    Think step by step and explain your reasoning.
    """

@marvin.fn
def chat(question:str, summary: list[dict[str, str]]) -> AIChatResponse:
    """
    The `summary` input provided contains list of incident data in the form of json created on incidents. With the summary information provide, answer the `question` below.
    Think step by step and explain yourself.
    question is - {{question}}
    """


@marvin.fn
def chat_for_text_response(question:str, summary: list[str]) -> AIChatResponseOnlyText:
    """
    The `summary` input provided contains list of incident data in the form of json created on incidents. With the summary information provide, answer the `question` below.
    Think step by step and explain yourself.
    question is - {{question}}
    """

@marvin.fn
def construct_timeline_classification_mttd_mttk(short: str, description: str, notes:str) -> Combined:
    """
    Timeline: Construct a timeline of actions from the below incident's short description: `short`, full description: `description`, and resolution notes: `notes`.
    We want to use this timeline to understand the sequence of events that occurred during the incident.
    We want to then help us learn and troubleshoot similar incidents in the future.

    classification: you an expert at root cause analysis.

    You perform root cause analysis using the following technique.

    You analyze any given issue across 3 separate areas.

    System: The software system that needs to run reliably. This includes your source code, your build release pipelines, your test scripts etc.

    Environment: The environment in which the system operates. This could include network conditions, user behavior, macroeconomic situation, general trends etc.

    Control: This is at the boundary of system and its environment. This could include infrastructure management, DNS, load balancing, capacity planning etc.

    System Vulnerability: These are weaknesses or flaws in a software system that could potentially be exploited.

    Worst Case Input: This refers to specific data or commands sent to the system that specifically trigger the vulnerability.

    System Failure: The result of the vulnerability being exploited.

    Root Cause: The process of identifying the fundamental reason for the system failure.

    Environment Interaction: This emphasizes that the vulnerability is often exposed through the system's interaction with its external environment.

    For the given `description` and `action_list`, classify the incident into one of System, Control, Environment.

    Think step by step and explain your reasoning.


    mttd: Time to detect is the time that has elapsed between when the incident first occurred to when it was first detected.


    mttk: Time to know is the time that has elased between when the incident was first detected to when a cause for the incident was identified.
    From the given `timeline`, calculate the time to know in minutes.
    Think carefully and step by step.

    recommendations: Give me top 1 recommendations based on timeline of actions, classification type and their respective reasoning. 
    Do not give genric recommendations like add monitoring, logging, patching etc.
    The business impact should state what metric this would change like MTTD, MTTK, MTTF etc by x% where relevant. Be specific and techinical. 

    maturityAnalysis: Based on the input containing incident timeline, please generate an input object for a maturity grid analysis. we also need the reasoning behind this and any improvements.
    The input should include categories ('Service Monitoring', 'Metrics SLIs / SLOs', 'Incident / Crisis Mgmt.', 'Build & Release Mgmt', 'Capacity Mgmt.', 'Secure/policy as code', 'Problem Mgmt.', 'Cost Optimization', 'Toil Reduction', 'BCDR') mapped to their respective maturity levels ('Discovering', 'Growth-Focused', or 'Industry-Leading').  Think step by step.
    """
    

#######################################