
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
import marvin
from typing import Any


#Classes
class ClassificationType(str, Enum):
    control = 'Control'
    system = 'System'
    environment = 'Environment'

class Classification(BaseModel):
    classification: ClassificationType
    reason: str


class Action(BaseModel):
    timestamp: datetime
    description: str

class DataFile(BaseModel):
    columnName: str
    value:str

class DataFileColumnDescriptions(BaseModel):
    ColumnName: str
    Description:str
    ContainsIncidentInformation:bool
    ContainsClassificationInformation:bool

class Recommendation(BaseModel):
    idea: str
    reason: str
    business_impact: str

class MaturityAnalysis(BaseModel):
    category: str
    maturityLevel: str
    reason: str

class TimetoKnowReason(BaseModel):
    time_to_know: float


class AIChatResponseOnlyText(BaseModel):
    response: str

@marvin.model(instructions='Always generate strings values')
class ChartData(BaseModel):
    """
    ChartData is a model representing the data required to generate a chart.

    Attributes:
        xAxisData (list[str]): A list of strings representing the data points for the x-axis.
        yAxisData (list[str]): A list of strings representing the data points for the y-axis.
        xAxislabel (str): A string representing the label for the x-axis.
        yAxislabel (str): A string representing the label for the y-axis.
    """
    xAxisData: list[Any]
    yAxisData: list[Any]
    xAxislabel:str
    yAxislabel:str

class AIChatResponse(BaseModel):
    response: str
    plotData:ChartData

class Combined(BaseModel):
    action: list[Action]
    classification: ClassificationType
    mttk:float
    mttd:float
    recommendations: list[Recommendation]
    maturityAnalysis: list[MaturityAnalysis]




