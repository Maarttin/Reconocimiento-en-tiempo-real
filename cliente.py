import asyncio
import websockets
import cv2
import numpy as np

# Dirección del servidor WebSocket
SERVER_URL = "ws://localhost:8000/ws"

async def receive_frames():
    async with websockets.connect(SERVER_URL) as websocket:
        # Crear una ventana de video y establecer su tamaño
        window_name = "YOLOv5 - Detección en Tiempo Real"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Establecer un tamaño personalizado para la ventana
        window_width, window_height = 1280, 720  # Tamaño de la ventana
        cv2.resizeWindow(window_name, window_width, window_height)

        while True:
            try:
                # Recibir los datos de la imagen desde el WebSocket
                frame_bytes = await websocket.recv()

                # Convertir los bytes de la imagen a un array de numpy
                np_arr = np.frombuffer(frame_bytes, np.uint8)

                # Decodificar la imagen
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Redimensionar el frame para que se ajuste al tamaño de la ventana
                frame_resized = cv2.resize(frame, (window_width, window_height))

                # Mostrar la imagen con OpenCV en una sola ventana
                cv2.imshow(window_name, frame_resized)

                # Salir si se presiona la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error: {e}")
                break

        # Cuando terminemos, cerramos la ventana
        cv2.destroyAllWindows()

# Ejecutar el cliente WebSocket
asyncio.run(receive_frames())
