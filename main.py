
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from fastapi.responses import JSONResponse
import librosa
from pydub import AudioSegment
import socketio
import uvicorn
import random


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


sio = socketio.AsyncServer(
    async_mode='asgi', logger=True, engineio_logger=True)
socket_app = socketio.ASGIApp(sio)

model = tf.lite.Interpreter(model_path="./model/advanced_model.tflite")

classes = [
    "air_conditioner",
    "ambulance",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "firetruck",
    "gun_shot",
    "jackhammer",
    "traffic",
]

# Routes.......................................................


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    response_data = await predict_audio(file)

    return JSONResponse(response_data)

connected_clients = []


@app.post("/remote_predict")
async def predict(
    file: UploadFile = File(...)
):
    print(connected_clients)
    response_data = await predict_audio(file)
    await sio.emit("prediction", response_data)
    return JSONResponse(response_data)


# Sockets.......................................................

app.mount("/", socket_app)


@sio.event
async def connect(sid, environ):
    connected_clients.append(sid)
    print('Connected:', sid)


@sio.event
async def disconnect(sid):
    connected_clients.remove(sid)
    print('Disconnected:', sid)


@sio.event
async def trigger(sid, data):
    scores = []

    for class_name in classes:
        if class_name == data:
            score = random.uniform(0.8, 1.0)
        else:
            score = random.uniform(0.0, 0.3)
        scores.append(score)

    response_data = {
        "class_scores":  [scores],
        "classes": data
    }
    await sio.emit("prediction", response_data)


# Functions.......................................................

async def predict_audio(file):
    print(file.filename)
    wav_file_path = await convert_audio(file)

    print(wav_file_path)
    waveform, sr = librosa.load(wav_file_path, sr=16000)
    if waveform.shape[0] % 16000 != 0:
        waveform = np.concatenate([waveform, np.zeros(16000)])

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.resize_tensor_input(input_details[0]['index'], (1, len(waveform)))
    model.allocate_tensors()

    model.set_tensor(input_details[0]['index'],
                     waveform[None].astype('float32'))
    model.invoke()

    class_scores = model.get_tensor(output_details[0]['index'])
    class_scores_list = class_scores.tolist()
    print(" ")
    print(" ")
    print("class_scores", class_scores.tolist())
    print(" ")
    print(" ")
    print("Class : ", classes[class_scores.argmax()])
    os.remove(wav_file_path)

    response_data = {
        "class_scores":  class_scores_list,
        "classes": classes[class_scores.argmax()]
    }

    return response_data


async def convert_audio(file: UploadFile):
    wav_path = ""
    temp_file_path = f"{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        content = await file.read()
        temp_file.write(content)

    if (file.filename.split(".")[1] != 'wav'):
        audio = AudioSegment.from_file(
            temp_file_path, format=file.filename.split(".")[1])
        wav_path = temp_file_path.replace(".mp4", ".wav")
        audio.export(wav_path, format="wav")
        os.remove(temp_file_path)

    else:
        wav_path = temp_file_path
    return wav_path

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    uvicorn.run("main:app", host='0.0.0.0', port=port, reload='true')
