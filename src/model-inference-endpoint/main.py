from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

import sys
sys.path.append('src/text_loader')

from text_loader.loader import DataLoader

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    # Clean + vectorize new tweet
    data_loader = DataLoader()
    #clf = mlflow.pyfunc.load_model("models:/tweet_classifier/Production")

    model = data_loader.train_model()
    party = data_loader.predict(model, [input_data.input_texts])

    return {"prediction": party[0]}

