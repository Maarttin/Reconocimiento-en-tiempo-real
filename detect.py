from fastapi import FastAPI, WebSocket
import torch
import cv2
import numpy as np
import asyncio


model = torch.hub.load('ultralytics/yolov5', 'custom', path="/home/marti/Documentos/object_detection/model/best.pt")


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Servidor YOLOv5 en ejecución"}

# Endpoint WebSocket para detección en tiempo real
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Iniciar la captura de video desde la cámara web
    cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

    try:
        while True:
            # Leer un frame de la cámara
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar la detección con YOLOv5 cada 0.2 segundos (~5 FPS)
            results = model(frame)
            rendered_frame = results.render()[0]

            # Convertir el frame procesado a formato JPEG
            _, buffer = cv2.imencode('.jpg', rendered_frame)
            frame_bytes = buffer.tobytes()

            # Enviar el frame procesado al cliente
            await websocket.send_bytes(frame_bytes)

            # Pausa de 0.2 segundos para controlar la tasa de frames
            await asyncio.sleep(0.2)  # Aproximadamente 5 FPS

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Liberar la cámara y cerrar la conexión
        cap.release()
        await websocket.close()

# Ejecutar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
