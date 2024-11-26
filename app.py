import numpy as np
import librosa
import librosa.display
import os
import warnings
from IPython.display import Audio
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from tensorflow.image import resize
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torchaudio
import io
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import keras
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

os.environ["KERAS_BACKEND"] = "tensorflow"

'''
  model = AutoModel.from_pretrained("Abdelhameid/musicgenre")
    tokenizer = AutoTokenizer.from_pretrained("Abdelhameid/musicgenre")
    return model
'''
app = FastAPI()

classes=['blues', 'classical','country', 'disco', 'hiphop', 'jazz', 'metal', 'pop','reggae','rock']

def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Abdelhameid/musicgenre",
            filename="musicgenre_classifier.keras"  
        )

        return keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise Exception(f"Failed to load model: {e}")

def load_and_preprocess_data(file_path,target_shape=(150,150)):
     data=[]
     try:
                  

                  audio_data,sample_rate = librosa.load(file_path,sr=None)

                  chunk_duration = 4
                  overlap_duration = 2

                  chunk_samples = chunk_duration * sample_rate
                  overlap_samples = overlap_duration * sample_rate

                  num_chunks = int(np.ceil((len(audio_data)-chunk_samples)/(chunk_samples-overlap_samples)))+1
                  for i in range(num_chunks):

                      start = i*(chunk_samples-overlap_samples)
                      end = start+chunk_samples

                      chunk = audio_data[start:end]

                      mel_spectrogram = librosa.feature.melspectrogram(y=chunk,sr=sample_rate)

                      mel_spectrogram = resize(np.expand_dims(mel_spectrogram,axis=-1),target_shape)

                      data.append(mel_spectrogram)
     except Exception as e:
                   HTTPException(status_code=400, detail=f"Error processing audio file: {e}")

     return np.array(data)

def model_predict(X_test):
    model = load_model()
    y_pred= model(X_test)
    pred_cats = np.argmax(y_pred,axis=1)
    unique_elements, count = np.unique(pred_cats,return_counts=True)
    max_count = np.max(count)
    max_elements = unique_elements[count==max_count]
    return max_elements[0]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Music Genre Classifier</title>
        </head>
        <body>
            <h1>Welcome to the Music Genre Classifier API</h1>
            <p>Use the POST method to send an audio file and get genre predictions.</p>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
       try:
              file_location = f"temp_files/{file.filename}"
              if not os.path.exists('temp_files'):
                 os.makedirs('temp_files')
              with open(file_location, "wb") as buffer:
                 

                 buffer.write(await file.read())

                 X_test = load_and_preprocess_data(file_location)

                 c_idx = model_predict(X_test)

                 return {"genre_prediction": classes[c_idx]}

       except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
       
       
       


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.0:8000", port=8000)



'''
run:
uvicorn app:app --reload

use render

'''
    
