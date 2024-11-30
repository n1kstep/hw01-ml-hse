import re
from typing import Optional
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import pandas as pd

from settings import CAR_BRANDS


def pydantic_model_to_df(model_instance: BaseModel) -> pd.DataFrame:
    return pd.DataFrame([jsonable_encoder(model_instance)])


def convert_strs(df) -> pd.DataFrame: 
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 0).astype(float)
    return df


def extract_car_brands(text: str) -> str:
    pattern = r'\b(' + '|'.join(re.escape(brand) for brand in CAR_BRANDS) + r')\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    matches = list(set(match.title() for match in matches))
    if len(matches) > 0:
        return matches[0]
    else:
        return "unknown"


def parse_torque(text: str) -> pd.Series:

    def convert_torque(value, unit) -> Optional[float]:
        if unit == "kgm":
            return float(value) * 9.80665
        elif unit == "nm":
            return float(value)
        return None
    
    text = str(text)
    torque_match = re.search(r"(\d+\.?\d*)\s*(nm|kgm)", text, re.IGNORECASE)
    rpm_match = re.search(
        r"(\d{1,3}(?:,\d{3}|\d*)?)\s*-\s*(\d{1,3}(?:,\d{3}|\d*)?)\s*rpm|(\d{1,4}(?:,\d{3}|\d*)?)\s*rpm", 
        text, 
        re.IGNORECASE
    )
    
    torque_value, rpm_max = 0, 0
    if torque_match:
        torque_value = convert_torque(torque_match.group(1), torque_match.group(2).lower())

    if rpm_match:
        if rpm_match.group(1) and rpm_match.group(2):
            rpm_max = int(rpm_match.group(2).replace(",", ""))
        elif rpm_match.group(3):
            rpm_max = int(rpm_match.group(3).replace(",", ""))

    return pd.Series([torque_value, rpm_max])
