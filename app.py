from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import io
from fastapi.responses import StreamingResponse

from settings import MODEL_PATH
from utils import pydantic_model_to_df, extract_car_brands, parse_torque, convert_strs


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


class ItemResponse(Item):
    prediction: float


class ItemsResponses(BaseModel):
    responses: List[ItemResponse]


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["ridge"] = joblib.load(MODEL_PATH)
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "name": "Cars selling price prediction",
        "description": "This is ML model for selling price prediction based on given car features",
    }


@app.post("/predict_item", response_model=ItemResponse)
async def predict_item(item: Item):
    df = pydantic_model_to_df(item)
    df = convert_strs(df)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(parse_torque)
    df['name'] = df['name'].apply(extract_car_brands)

    score = ml_models["ridge"].predict(df)[0]
    response = item.model_dump() | {"prediction": score}
    return response


@app.post("/predict_items", response_model=ItemsResponses)
async def predict_items(items: List[Item]):
    df = pd.concat([pydantic_model_to_df(item) for item in items])
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(parse_torque)
    df['name'] = df['name'].apply(extract_car_brands)
    df = convert_strs(df)

    scores = ml_models["ridge"].predict(df)
    responses = [item.model_dump() | {"prediction": float(score)} for item, score in zip(items, scores)]
    return {"responses": responses}

from fastapi import File, UploadFile, HTTPException


@app.post("/predict_items_csv")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()

    df = pd.read_csv(file.filename)
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(parse_torque)
    df['name'] = df['name'].apply(extract_car_brands)
    df = convert_strs(df)
    df["prediction"] = ml_models["ridge"].predict(df)

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response

