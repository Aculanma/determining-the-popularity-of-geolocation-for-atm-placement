from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
import io
import traceback

app = FastAPI()

# Подгружаем модель и OneHotEncoder
model_path = os.path.join(os.path.dirname(__file__), "model_weights.pkl")
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

ohe_path = os.path.join(os.path.dirname(__file__), "onehotencoder.pkl")
with open(ohe_path, 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)


# Pydantic модель для валидации данных
class CSVRecord(BaseModel):
    lat: float
    long: float
    settlement: str
    street_name: str
    settlement_count: int
    atm_group: str
    postal_code: str

# Для валидации списка записей
class CSVData(BaseModel):
    records: List[CSVRecord]


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Преобразуем типы данных
        data['settlement'] = data['settlement'].astype(str)
        data['lat'] = data['lat'].astype(float)
        data['long'] = data['long'].astype(float)
        data['settlement_count'] = data['settlement_count'].astype(int)
        data['atm_group'] = data['atm_group'].astype(str)
        data['postal_code'] = data['postal_code'].astype(str)

        # Категориальные признаки
        cat_features = ['settlement', 'street_name', 'atm_group', 'postal_code']

        # Преобразуем с помощью OHE
        encoded = ohe.transform(data[cat_features]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_features), index=data.index)

        # Объединяем обработанные данные
        return pd.concat([data.drop(columns=cat_features), encoded_df], axis=1)

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Ошибка обработки данных: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение файла
        data = file.file.read()
        df = pd.read_csv(io.BytesIO(data), encoding='windows-1251')

        # Валидация данных с помощью Pydantic
        try:
            records = [CSVRecord(**row) for row in df.to_dict(orient='records')]
            csv_data = CSVData(records=records)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ошибка в данных CSV: {str(e)}")

        # Предобработка данных
        processed_data = preprocess_data(df)

        # Предсказания
        predictions = loaded_model.predict(processed_data)
        df['predicted_index'] = predictions

        # Генерация ответа в виде CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(e)}")