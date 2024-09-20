import os
import csv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib
import scipy
from scipy.signal import spectrogram
from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse

# Check TensorFlow version
if tf.__version__ != '2.11.0':
    print(f"Warning: Current TensorFlow version is {tf.__version__}. Model was trained with 2.11.0.")

matplotlib.use('Agg')

app = FastAPI()

# AI Model
class AIModel:
    def __init__(self):
        try:
            # Try to load the model normally
            self.model = keras.models.load_model("./models/model.h5")
        except ValueError as e:
            if 'DepthwiseConv2D' in str(e):
                # If DepthwiseConv2D is the issue, try to load with custom objects
                from tensorflow.keras.layers import DepthwiseConv2D
                self.model = keras.models.load_model("./models/model.h5", 
                                                     custom_objects={'DepthwiseConv2D': DepthwiseConv2D})
            else:
                # If it's a different error, try to provide more information
                print(f"Error loading model: {str(e)}")
                print("Current TensorFlow version:", tf.__version__)
                print("CUDA available:", tf.test.is_built_with_cuda())
                print("GPU available:", tf.config.list_physical_devices('GPU'))
                raise

    def predict(self, img_path: str):
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224, 3))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        return predictions

model = AIModel()

# Tools
def refine(file_path: str):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    rows = rows[14:]
    rows = [row for i, row in enumerate(rows) if i % 2 != 0]
    rows = rows[1500:]

    for i in range(len(rows)):
        for j in range(1, len(rows[i])):
            rows[i][j] = rows[i][j-1] if j > 0 else ''

    for i in range(len(rows)):
        rows[i].insert(1, str(float(rows[i][0]) * 3))
        rows[i][0] = str(i * 0.01)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def spec(csv_file):
    df = pd.read_csv(csv_file)
    time = df.iloc[:, 0].values
    frequency = df.iloc[:, 1].values
    fs = 1 / (time[1] - time[0])

    nperseg = 447
    window = scipy.signal.windows.blackman(nperseg)
    noverlap = nperseg - 1 - 224
    f, t, Sxx = spectrogram(frequency, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    Sxx = np.log(np.abs(Sxx)**2 + 1e-10)

    matplotlib.pyplot.figure(figsize=(7/3, 7/3), facecolor='none', dpi=96)
    matplotlib.pyplot.pcolormesh(t, f, Sxx)
    matplotlib.pyplot.axis('off')
    output_image_path = f"{csv_file}.png"
    matplotlib.pyplot.savefig(output_image_path, transparent=True, bbox_inches=0, pad_inches=0)
    matplotlib.pyplot.close()

# API Routes
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")
        file_location = f"temp_files/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        return JSONResponse(content={"message": "File uploaded successfully", "file_location": file_location})
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

@app.post("/run_model/{file_location}")
async def run_model(file_location: str):
    try:
        file_location = f"temp_files/{file_location}"
        refine(f"{file_location}.csv")
        spec(f"{file_location}.csv")
        prediction = model.predict(f"{file_location}.csv.png")
        
        os.remove(f"{file_location}.csv.png")
        os.remove(f"{file_location}.csv")

        return JSONResponse(content={"message": "model run successfully", "prediction": str(prediction[0][0])}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    
    # Print environment information
    print("TensorFlow version:", tf.__version__)
    print("CUDA available:", tf.test.is_built_with_cuda())
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
