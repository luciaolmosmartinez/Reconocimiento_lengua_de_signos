import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

from modelo import usar_modelo

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
BaseOptions = mp.tasks.BaseOptions

BASE_DIR = Path(__file__).resolve().parent

HAND_LANDMARKER_PATH = BASE_DIR / "hand_landmarker.task"
CSV_PATH = BASE_DIR / "dataset_tfg.csv"

# Guardar los puntos en el archivo CSV
def guardar_datos(resultados, etiqueta, archivo_csv):
    fila = [etiqueta] + list(resultados) # La fila contiene la etiqueta asignada y los puntos resultantes
    # Guardamos en el CSV, que debe contener una primera fila con el nombre de cada casilla
    with open(archivo_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fila)

# Dibujar landmarks (manual con cv2)
def dibujar_landmarks_hand(result, image, frame):
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * image.width)
                y = int(landmark.y * image.height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

# Normalización de los puntos
def calcular_escala(resultados):
    lista_puntos = []
    for punto in resultados:
        lista_puntos.append([punto.x, punto.y, punto.z])

    puntos = np.array(lista_puntos)
    muneca = puntos[0]
    centrados = puntos - muneca

    distancias = np.linalg.norm(centrados, axis=1) # Distancia euclidea de cada punto al (0,0,0)
    maxima = np.max(distancias)

    if maxima == 0:
        maxima = 1
    
    normalizados = centrados/maxima

    return normalizados.flatten()

# VIDEOCAMARA
def videoCamara(modelo):

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): # Si la cámara no se ha podido abrir
        print("Cannot open camera")
        exit()
    
    opciones_manos = HandLandmarkerOptions( # Opciones para la detección de las manos
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
        running_mode=RunningMode.VIDEO, # Para que tome la imagen de la cámara
        num_hands=2 # Para que detecte las dos manos
    )

    hand_landmarker = HandLandmarker.create_from_options(opciones_manos)

    print("Para poder cerrar el programa, pulse \"q\" mientras tiene la cámara seleccionada")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se puede recibir el frame.")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        resultados_hand = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        dibujar_landmarks_hand(resultados_hand,mp_image,frame)

        prediccion_letra = "Ninguna"
        
        if resultados_hand.hand_landmarks:
            mano_detectada = resultados_hand.hand_landmarks[0]
            
            datos_mano_escalados = calcular_escala(mano_detectada)

            prediccion_letra = usar_modelo(datos_mano_escalados, modelo)

            if prediccion_letra is not None:
                cv2.putText(frame, f"Prediccion IA: {prediccion_letra.upper()}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Prediccion IA: modelo no cargado", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'): # Salir
            break
    cap.release()
    cv2.destroyAllWindows()

# IMAGEN
def imagen(ruta_carpeta, etiqueta, ver_imagen):
    opciones_manos = HandLandmarkerOptions( # Opciones para la detección de las manos
        base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
        running_mode=RunningMode.IMAGE, # Para que tome la imagen que se abra
        num_hands=1, # Para que detecte una
        # Para que pueda reconocer más imágenes y se puede realizar la selección manualmente
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1
    )

    hand_landmarker = HandLandmarker.create_from_options(opciones_manos)

    for archivo in os.listdir(ruta_carpeta):
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        frame = cv2.imread(ruta_imagen)
        print('imagen que se guarda: ', ruta_imagen)

        if frame is None:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        resultados_hand = hand_landmarker.detect(mp_image)

        # Código para mostrar las imágenes una a una
        if resultados_hand.hand_landmarks and ver_imagen == 's':
            # Esta línea dibujará el esqueleto sobre la foto
            dibujar_landmarks_hand(resultados_hand, mp_image, frame)
            cv2.imshow("¿Que ve la IA?", frame)
            print('Pulse cualquier tecla mientras tiene la imagen seleccionada para pasar a la siguiente.')
            cv2.waitKey(0)

        for mano in resultados_hand.hand_landmarks:
            for id, landmark in enumerate(mano):
                x_norm = landmark.x 
                y_norm = landmark.y
                z_norm = landmark.z
                print('coordenadas punto ', id , ': x=', x_norm, '; y=', y_norm, '; z=', z_norm, '\n')
            resultados_escalados = calcular_escala(mano)
            guardar_datos(resultados_escalados, etiqueta, CSV_PATH)

def main():
    print("========================================")
    print("           LENGUAJE DE SIGNOS           ")
    print("========================================")
    print("Seleccione una opción:")
    print("[c] - Reconocer vídeo de la CÁMARA")
    print("[i] - Reconocer IMAGEN desde ruta y almacenar puntos")
    print("[q] - Salir")
    print("----------------------------------------")

    # Usamos un bucle para esperar la decisión del usuario
    while True:
        opcion = input("Introduzca su elección: ").lower()

        if opcion == 'c':
            print("Iniciando cámara...")
            modelo = input("¿Qué modelo quiere emplear? (arbol/bosque/red neuronal/knn/svm) ").lower()
            videoCamara(modelo)
            break
            
        elif opcion == 'i':
            ruta = BASE_DIR / input("Introduzca el nombre de la carpeta: ")
            etiqueta = input("Introduzca la etiqueta con la que guardar los datos: ")
            ver_imagen = input("¿Quiere ver todas las imágenes que se procesen? (s/n): ").lower()
            imagen(ruta, etiqueta, ver_imagen)
            break
            
        elif opcion == 'q':
            print("Saliendo...")
            exit()
            
        else:
            print("Opción no válida. Inténtelo de nuevo.")

# PRINCIPIO DEL CODIGO
if __name__ == "__main__":
    main()