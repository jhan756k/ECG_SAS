from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
from tools import tdrefine, tdspectrogram
from models.ai import model

router = APIRouter()

@router.post("/run_model/{file_location}")
async def run_model(file_location: str):
    try:
        file_location = f"temp_files/{file_location}"
        tdrefine.refine(f"{file_location}.csv")
        tdspectrogram.spec(f"{file_location}.csv")
        prediction = model.predict(f"{file_location}.csv.png")
        
        os.remove(f"{file_location}.csv.png")
        os.remove(f"{file_location}.csv")

        return JSONResponse(content={"message": "model run successfully", "prediction": str(prediction[0][0])})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)