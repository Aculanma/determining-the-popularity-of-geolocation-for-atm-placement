from model import preprocess_data
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
import io
import uvicorn
import logging as log

file_log = log.FileHandler('server_log.log')
console_out = log.StreamHandler()

log.basicConfig(handlers=(file_log, console_out), 
                    format='[%(asctime)s | %(levelname)s]: %(message)s', 
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=log.INFO)

app = FastAPI()

# Подгружаем модель и OneHotEncoder
model_path = os.path.join(os.path.dirname(__file__), "resources/model_weights.pkl")
with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

ohe_path = os.path.join(os.path.dirname(__file__), "resources/onehotencoder.pkl")
with open(ohe_path, 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)

@app.get("/api/v1/healthcheck/")
def healthcheck():
    return 'Health - OK'

@app.post("/api/v1/predict/")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    try:
        # Чтение файла
        data = file.file.read()
        df = pd.read_csv(io.BytesIO(data), encoding='windows-1251')

        # Валидация данных с помощью Pydantic
        try:
            log.info("Validating CSV with Pydantic")
            records = [CSVRecord(**row) for row in df.to_dict(orient='records')]
            csv_data = CSVData(records=records)
        except Exception as e:
            log.error("Exception while validating CSV", e)
            raise HTTPException(status_code=422, detail=f"Ошибка в данных CSV: {str(e)}")

        # Предобработка данных
        processed_data = preprocess_data(df, ohe)
        log.debug("Data processed")

        # Предсказания
        predictions = loaded_model.predict(processed_data)
        df['predicted_index'] = predictions
        log.info("Predictions = [%s]", predictions)

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
        log.error("Exception pd.errors.ParserError", e)
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except ValueError as e:
        log.error("Exception ValueError", e)
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(e)}")
    

# Pydantic модель для валидации данных
class CSVRecord(BaseModel):
    lat: float
    long: float
    settlement: str
    street_name: str
    settlement_count: int
    atm_group: int
    postal_code: int

# Для валидации списка записей
class CSVData(BaseModel):
    records: List[CSVRecord]

if __name__ == "__main__":
    port = int(os.getenv('BACKEND_PORT'))
    log.info("Starting web-server with port = [%s]", port)
    uvicorn.run(app, host="0.0.0.0", port=port)