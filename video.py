from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
import shutil
import os
from pathlib import Path
from fastapi.responses import StreamingResponse
import io
import time


model = torch.hub.load('ultralytics/yolov5', 'custom', path="/home/marti/Documentos/object_detection/model/best.pt")


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Servidor YOLOv5 en ejecución"}

TEMP_VIDEO_PATH = "temp_video.mp4"

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):#le indica a FastAPI que file es un archivo que se enviará en la solicitud HTTP usando multipart/form-data.
    try:
        
        with open(TEMP_VIDEO_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)#copia el archivo temporalmente en el servidor

        return {"message": "Video subido con éxito, disponible en /video-stream/"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/video-stream/")
async def video_stream():
    def generate():
        cap = cv2.VideoCapture(TEMP_VIDEO_PATH)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar cada frame con YOLOv5
            results = model(frame)
            rendered_frame = results.render()[0]

            # Convertir frame a formato JPEG
            _, buffer = cv2.imencode('.jpg', rendered_frame)
            frame_bytes = buffer.tobytes()

            # Enviar el frame como parte del stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            #time.sleep(0.2)

        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = f"processed_{Path(video_path).stem}.mp4"
    
    # Obtener propiedades del video original
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Definir el codec y crear un objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar la detección con YOLOv5
        results = model(frame)
        rendered_frame = results.render()[0]

        # Escribir el frame procesado en el video de salida
        out.write(rendered_frame)

    cap.release()
    out.release()

    return output_path
