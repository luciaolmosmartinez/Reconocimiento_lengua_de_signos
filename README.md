# Reconocimiento de Lengua de Signos

**Nombre del proyecto:** Reconocimiento de Lengua de Signos

**Tutor:** Ginés García Mateos

**Autora:**
1. Lucía Olmos Martínez

**Breve descripción:**  
Este proyecto corresponde al Trabajo Fin de Grado y tiene como objetivo desarrollar un sistema de reconocimiento de lengua de signos mediante técnicas de visión artificial y aprendizaje automático. La aplicación es capaz de detectar la posición de la mano utilizando MediaPipe y clasificar los gestos mediante diferentes modelos de Machine Learning previamente entrenados.

Se han entrenado y evaluado distintos algoritmos de clasificación, entre ellos:
- K-Nearest Neighbors (KNN)
- Árboles de decisión
- Random Forest
- Máquinas de Vectores de Soporte (SVM)
- Redes neuronales

Los modelos entrenados se encuentran almacenados en la carpeta:

`modelos/`

La detección de los puntos clave de la mano se realiza mediante el modelo de MediaPipe:

`hand_landmarker.task`

## Ejecución de la aplicación

### Reconocimiento en tiempo real

Para iniciar el reconocimiento de gestos mediante la cámara web, ejecutar:

```bash
python video.py
```

Es necesario que el archivo:

```
hand_landmarker.task
```

se encuentre en la misma carpeta que `video.py`.

### Evaluación de los modelos

Para ejecutar el script encargado de cargar y evaluar los modelos entrenados, ejecutar:

```bash
python modelo.py
```

Los modelos previamente entrenados deben encontrarse en la carpeta:

```
modelos/
```

## Documentación

La memoria completa del Trabajo Fin de Grado puede consultarse en:

```
Memoria_TFG_Lucia_Olmos_Martinez.pdf
```

## Estructura del proyecto

```
Reconocimiento_lengua_de_signos/
│
├── video.py                  # Reconocimiento en tiempo real mediante webcam
├── modelo.py                 # Evaluación y utilización de los modelos entrenados
├── hand_landmarker.task      # Modelo de MediaPipe para la detección de manos
├── modelos/
│   ├── decision_tree.pckl
│   ├── knn.pckl
│   ├── random_forest.pckl
│   ├── red_neuronal.pckl
│   └── svm.pckl
└── README.md
```
