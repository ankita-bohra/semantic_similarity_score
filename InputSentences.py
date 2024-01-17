from pydantic import BaseModel
#1. Class which describes Text1 and Text2

class InputSentences(BaseModel):
    sentence1: str
    sentence2: str

class MyResponse(BaseModel):
    result: list[float]