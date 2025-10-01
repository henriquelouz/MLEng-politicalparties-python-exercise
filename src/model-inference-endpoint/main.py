from fastapi import FastAPI
from pydantic import BaseModel

import sys
sys.path.append('src/text_loader')

from text_loader.loader import DataLoader

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    data_loader = DataLoader()
    party = data_loader.predict([input_data.input_texts])

    return {"prediction": party[0]}

