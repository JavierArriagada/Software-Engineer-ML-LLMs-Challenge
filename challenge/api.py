import fastapi
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List
from model import DelayModel  # Import your DelayModel from challenge.py

app = fastapi.FastAPI()

# Define a Pydantic model to validate the request JSON data
class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

# Create an instance of the DelayModel
delay_model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Endpoint to check the health of the API.
    """
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(flights: List[FlightData]) -> dict:
    """
    Endpoint to predict delays for flights.

    Args:
        flights (List[FlightData]): List of flight data.

    Returns:
        dict: Predicted delays.
    """
    # Create a DataFrame from the input data
    data = pd.DataFrame([flight.dict() for flight in flights])

    # Preprocess the data using the DelayModel's preprocess method
    features = delay_model.preprocess(data)

    # Use the DelayModel to make predictions
    predictions = delay_model.predict(features)

    return {"predict": predictions}