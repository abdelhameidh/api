# -*- coding: utf-8 -*-
"""Musicgen.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EXHLm-TUZg-DoQL58LQIzzGzzvDamPr7
"""

!pip install git+https://github.com/facebookresearch/audiocraft#egg=audiocraft
!pip install soundfile

!pip install uvicorn

!pip install fastapi
!pip install pickle5
!pip install pydantic
!pip install scikit-learn
!pip install requests
!pip install pypi-json
!pip install nest-asyncio

!pip install pyngrok

"""## modified version

##An API that plays the generated music

##Use this cell
"""

ngrok.set_auth_token("2lqAukpXuMqPJMTbcmmH2A0MiIs_73qvqn7CJkuhcRReFhQ5h")

from fastapi.responses import FileResponse
import io
import torch
import tempfile
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import uuid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=8)

class MusicGenerationRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 8
    model_size: Optional[str] = 'small'

@app.post("/generate-music")
async def generate_music(request: MusicGenerationRequest):
    try:
        model = MusicGen.get_pretrained(request.model_size)
        model.set_generation_params(duration=request.duration)

        wav = model.generate([request.prompt])

        audio_data = wav[0].to(dtype=torch.float32).to(device)
        generated_duration = audio_data.shape[0] / model.sample_rate

        filename = f"generated_{uuid.uuid4()}.wav"
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(output_dir, filename)

        audio_write(
            output_path.split('.')[0],
            audio_data,
            model.sample_rate,
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "X-Audio-Duration": f"{generated_duration:.2f} seconds",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    config = Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = Server(config=config)
    loop = asyncio.get_event_loop()
    nest_asyncio.apply()
    loop.create_task(server.serve())

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

