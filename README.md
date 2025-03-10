Proyecto Phygital Human Bone
Descripción General
Este proyecto se centra en el análisis automatizado de imágenes de microtomografía CT para la detección y análisis de canales de Havers en tejido óseo cortical. Utilizando técnicas de visión por computadora y aprendizaje automático, el software permite identificar estructuras microanatómicas del hueso y analizar su distribución espacial.
Componentes Principales
El proyecto consta de dos scripts principales:

fixed-phygital-code.py: Aplicación para la detección automática de canales de Havers utilizando un modelo YOLO entrenado.
cuadrantes-analyzer.py: Herramienta complementaria que analiza la distribución espacial de los canales dividiendo la imagen en nueve cuadrantes.

Tecnologías Utilizadas

Python como lenguaje de programación principal
OpenCV para el procesamiento de imágenes
YOLO (You Only Look Once) para la detección de objetos
Pandas y NumPy para el análisis de datos
Matplotlib para la visualización de resultados
Tkinter para la interfaz gráfica de usuario

Funcionalidades
fixed-phygital-code.py

Carga y procesamiento de imágenes de microtomografía CT
División de imágenes en segmentos más pequeños para análisis detallado
Detección automática de canales de Havers mediante modelo YOLO
Cálculo de parámetros relevantes: posición, área y distribución de canales
Generación de visualizaciones: mapa de coordenadas y mapa de calor
Interfaz gráfica para interacción con el usuario

cuadrantes-analyzer.py

División de la imagen analizada en nueve cuadrantes (matriz 3x3)
Análisis de la distribución de canales por cuadrante
Identificación del cuadrante con mayor densidad de canales
Visualización de resultados con marcado de cuadrantes
Exportación de datos a formato Excel

Proceso de Análisis

Selección de imagen: El usuario carga una imagen de microtomografía CT
Procesamiento: La imagen se divide en segmentos más pequeños
Detección: Se aplica el modelo YOLO para identificar canales de Havers
Análisis: Se calculan estadísticas relevantes (conteo, áreas, distancias)
Visualización: Se generan gráficos y mapas de calor para facilitar la interpretación
Análisis espacial: Se estudia la distribución por cuadrantes (opcional)

Configuración y Requisitos
El proyecto requiere las siguientes bibliotecas de Python:

opencv-python (cv2)
numpy
pandas
matplotlib
pillow (PIL)
ultralytics (YOLO)
torch

Modelo de Machine Learning
El sistema utiliza un modelo YOLO entrenado específicamente para la detección de canales de Havers en imágenes histológicas. El entrenamiento se realizó con los siguientes parámetros:

Modelo base: YOLOv8n
Epochs: 100
Batch size: 16
Tamaño de imagen: 640x640
Threshold de confianza: 0.4

Datos de Entrada/Salida
Entrada:

Imágenes de microtomografía CT en formato JPG, JPEG o PNG

Salida:

Archivo Excel con coordenadas y áreas de los canales detectados
Visualizaciones gráficas (mapa de coordenadas, mapa de calor)
Imagen con cuadrantes analizados (opcional)
Estadísticas por cuadrante (opcional)

Limitaciones

El rendimiento óptimo se obtiene con imágenes de microtomografía CT de alta calidad
Imágenes muy grandes pueden requerir redimensionamiento

Uso Recomendado
Este software está diseñado para:

Investigación en biomecánica y estructura ósea
Análisis histológico cuantitativo
Estudios de distribución espacial de canales de Havers
Desarrollo de modelos biomiméticos de tejido óseo
