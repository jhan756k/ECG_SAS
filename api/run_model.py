from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import os 
from tools import tdrefine, tdspectrogram
from models.ai import model

router = APIRouter()
results = {}

@router.post("/run_model/{file_location}")
async def run_model(file_location: str, background_tasks: BackgroundTasks):
    try:
        df = pd.read_csv(f"temp_files/{file_location}")
        background_tasks.add_task(process_file, f"temp_files/{file_location}", df)

        return JSONResponse(content={"message": "Model is running in the background"})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

def process_file(file_location: str, dataframe: pd.DataFrame):
    try:
        tdrefine.refine(dataframe)
        tdspectrogram.spec(file_location)
        prediction = model.predict(f"{file_location}.png")
        results[file_location] = prediction.tolist()
        
        os.remove(file_location)
        os.remove(f"{file_location}.png")
        os.remove(f"{file_location}.csv")
    except Exception as e:
        print(f"Error processing file: {e}")