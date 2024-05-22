from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import os

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
  try:
    file_location = f"temp_files/{file.filename}"
    async with open(file_location, "wb+") as file_object:
      await file_object.write(file.file.read())

    return JSONResponse(content={"message": "File uploaded successfully", "file_location": file_location})
  except Exception as e:
    return JSONResponse(content={"message": str(e)}, status_code=500)
