from fastapi import FastAPI
from api import upload, run_model

app = FastAPI()

app.include_router(upload.router)
app.include_router(run_model.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
