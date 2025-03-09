from PIL import Image
import os
import sys

def convert_tif_to_png(input_folder, output_folder, max_size_mb=3.5):  # Reducimos a 3.5MB para estar seguros
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Obtener la lista de archivos .tif en la carpeta de entrada
    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    
    print(f"Encontrados {len(tif_files)} archivos .tif para convertir")
    
    for i, tif_file in enumerate(tif_files):
        try:
            # Ruta completa del archivo de entrada
            input_path = os.path.join(input_folder, tif_file)
            
            # Nombre del archivo de salida (cambiar extensión a .png)
            output_filename = os.path.splitext(tif_file)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # Abrir la imagen
            with Image.open(input_path) as img:
                # Obtener dimensiones originales
                orig_width, orig_height = img.size
                
                # Convertir a RGB si es necesario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Comenzar con una reducción base del 60% para todas las imágenes
                new_width = int(orig_width * 0.6)
                new_height = int(orig_height * 0.6)
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Guardar con alta compresión
                resized_img.save(output_path, 'PNG', optimize=True, compress_level=9)
                
                # Verificar el tamaño y seguir reduciendo si es necesario
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                scale_factor = 0.9  # Reducción adicional del 10% cada vez
                
                while size_mb > max_size_mb:
                    new_width = int(new_width * scale_factor)
                    new_height = int(new_height * scale_factor)
                    resized_img = resized_img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img.save(output_path, 'PNG', optimize=True, compress_level=9)
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                # Calcular el porcentaje de reducción
                reduction_percent = (new_width / orig_width) * 100
                
                print(f"Convertido {i+1}/{len(tif_files)}: {tif_file} → {output_filename} ({size_mb:.2f} MB, {reduction_percent:.1f}% del tamaño original)")
            
        except Exception as e:
            print(f"Error al convertir {tif_file}: {e}")

# Ejemplo de uso
input_folder = r"C:\Users\joanb\OneDrive\Escritorio\TFG\AdManBio Teams\Histology Femur - IMAGES\HISTO_1"
output_folder = r"C:\Users\joanb\OneDrive\Escritorio\TFG\AdManBio Teams\Histology Femur - IMAGES\HISTO_1_PNG"
convert_tif_to_png(input_folder, output_folder)