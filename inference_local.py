from roboflow import Roboflow
import cv2
import os
import time

# Configura tu API key de Roboflow - Reemplaza con tu clave real
API_KEY = "MRdWG5zr7plOxlPk4ZkU"  # Reemplaza esto por tu clave

# Rutas de las carpetas
input_folder = r"C:\Users\joanb\OneDrive\Escritorio\TFG\AdManBio Teams\Histology Femur - IMAGES\HISTO_1_PNG"
output_folder = r"C:\Users\joanb\OneDrive\Escritorio\TFG\AdManBio Teams\Histology Femur - IMAGES\RESULTADOS_DETECCION"

# Configuración del modelo
PROJECT_ID = "tfg-dnnqi"  # Reemplaza con tu ID de proyecto
MODEL_VERSION = 1  # Reemplaza con tu número de versión

# Crear carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializa Roboflow
print("Inicializando Roboflow...")
rf = Roboflow(api_key=API_KEY)

# Lista todos los proyectos disponibles (útil para verificar)
print("Listando proyectos disponibles:")
workspace = rf.workspace()
projects = workspace.projects()
for project in projects:
    print(f"- {project}")

# Carga el modelo
print(f"Cargando el modelo (Proyecto: {PROJECT_ID}, Versión: {MODEL_VERSION})...")
try:
    model = workspace.project(PROJECT_ID).version(MODEL_VERSION).model
    
    # Configurar para inferencia local
    print("Configurando para inferencia local...")
    model.configure(confidence=40, overlap=30)
    model.use_local()
    
    print("Modelo cargado correctamente!")
    
    # Procesar cada imagen
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Encontradas {len(image_files)} imágenes para procesar")
    
    for i, filename in enumerate(image_files):
        print(f"\nProcesando imagen {i+1}/{len(image_files)}: {filename}")
        image_path = os.path.join(input_folder, filename)
        
        # Realizar predicción
        try:
            start_time = time.time()
            # Hacer la predicción
            result = model.predict(image_path, confidence=40, overlap=30).json()
            elapsed_time = time.time() - start_time
            
            print(f"  Predicción completada en {elapsed_time:.2f} segundos")
            print(f"  Detectados {len(result['predictions'])} canales de Havers")
            
            # Visualizar las predicciones
            if len(result['predictions']) > 0:
                image = cv2.imread(image_path)
                
                # Dibujar cada predicción
                for prediction in result["predictions"]:
                    x1 = prediction["x"] - prediction["width"] / 2
                    y1 = prediction["y"] - prediction["height"] / 2
                    x2 = prediction["x"] + prediction["width"] / 2
                    y2 = prediction["y"] + prediction["height"] / 2
                    
                    # Convertir a enteros
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Dibujar rectángulo
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Añadir texto con la confianza
                    confidence = prediction["confidence"]
                    label = f"{confidence:.2f}"
                    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Guardar imagen con predicciones
                output_path = os.path.join(output_folder, f"pred_{filename}")
                cv2.imwrite(output_path, image)
                print(f"  Imagen guardada en: {output_path}")
            else:
                print("  No se detectaron objetos en esta imagen")
                
        except Exception as e:
            print(f"  Error en la predicción: {e}")
    
    print("\nProcesamiento completo!")
    
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Verifica que el ID del proyecto y la versión sean correctos")