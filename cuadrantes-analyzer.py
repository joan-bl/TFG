import cv2
import os
import pandas as pd
import numpy as np
from tkinter import Tk, Button, Text, Frame, Label, ttk, filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def configure_window(window, title):
    """Configura el aspecto visual de la ventana principal"""
    window.title(title)
    window.geometry("1000x1000")
    window.configure(bg='#001f3f')

def configure_button(button):
    """Configura el aspecto visual de los botones"""
    button.configure(
        bg="#4CAF50",
        fg="white",
        font=("Helvetica", 12),
        padx=10,
        pady=5,
        relief="raised",
        borderwidth=2
    )
    # Añadir efectos hover
    button.bind("<Enter>", lambda e: button.configure(bg="#45a049"))
    button.bind("<Leave>", lambda e: button.configure(bg="#4CAF50"))

def reconstruir_imagen_con_detecciones(imagen_original, excel_path, output_path):
    """
    Reconstruye la imagen original con las detecciones de canales marcadas.
    
    Args:
        imagen_original: Ruta a la imagen original
        excel_path: Ruta al archivo Excel con las coordenadas de detecciones
        output_path: Ruta donde guardar la imagen reconstruida
    
    Returns:
        imagen_reconstruida: Imagen con las detecciones marcadas
    """
    # Cargar la imagen original
    imagen = cv2.imread(imagen_original)
    
    # Cargar datos de detecciones
    df = pd.read_excel(excel_path)
    
    # Dibujar cada canal detectado como un círculo en la imagen
    for i, row in df.iterrows():
        # Procesar valores de centro que podrían ser tensores o strings
        try:
            # Si es string que contiene tensor, extraer el número
            if isinstance(row['Center X'], str) and 'tensor' in row['Center X']:
                # Extraer el número entre paréntesis
                valor = row['Center X'].split('(')[1].split(')')[0]
                centro_x = float(valor)
            else:
                centro_x = float(row['Center X'])
            
            if isinstance(row['Center Y'], str) and 'tensor' in row['Center Y']:
                # Extraer el número entre paréntesis
                valor = row['Center Y'].split('(')[1].split(')')[0]
                centro_y = float(valor)
            else:
                centro_y = float(row['Center Y'])
                
            # Convertir a enteros para coordenadas de píxeles
            centro_x = int(centro_x)
            centro_y = int(centro_y)
            
            # Procesar el área de manera similar
            if isinstance(row['Ellipse Area (pixels^2)'], str) and 'tensor' in row['Ellipse Area (pixels^2)']:
                # Extraer el número entre paréntesis
                valor = row['Ellipse Area (pixels^2)'].split('(')[1].split(')')[0]
                area = float(valor)
            else:
                area = float(row['Ellipse Area (pixels^2)'])
                
            # Calcular radio aproximado del canal basado en el área (A = πr²)
            radio = int(np.sqrt(area / np.pi))
            
            # Dibujar círculo en la posición del canal
            cv2.circle(imagen, (centro_x, centro_y), radio, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error procesando fila {i}: {e}")
            continue  # Saltar esta detección si hay error
        
        # Dibujar círculo en la posición del canal
        cv2.circle(imagen, (centro_x, centro_y), radio, (0, 255, 0), 2)
    
    # Guardar imagen reconstruida
    cv2.imwrite(output_path, imagen)
    
    return imagen

def analizar_cuadrantes(imagen, df, output_path):
    """
    Divide la imagen en 9 cuadrantes y analiza la distribución de canales.
    
    Args:
        imagen: Imagen reconstruida con detecciones
        df: DataFrame con coordenadas y áreas de canales
        output_path: Ruta donde guardar la imagen final
    
    Returns:
        imagen_con_cuadrantes: Imagen con cuadrantes y marcado el de mayor densidad
        areas_por_cuadrante: Array con áreas por cuadrante
        canales_por_cuadrante: Lista de canales agrupados por cuadrante
    """
    height, width = imagen.shape[:2]
    
    # Calcular dimensiones de cuadrantes
    cuad_height = height // 3
    cuad_width = width // 3
    
    # Crear estructura para almacenar áreas por cuadrante
    areas_por_cuadrante = np.zeros(9)
    canales_por_cuadrante = [[] for _ in range(9)]
    
    # Clasificar cada canal en su cuadrante correspondiente
    for i, row in df.iterrows():
        try:
            # Procesar coordenadas X e Y que podrían ser tensores o strings
            if isinstance(row['Center X'], str) and 'tensor' in row['Center X']:
                # Extraer el número entre paréntesis
                valor = row['Center X'].split('(')[1].split(')')[0]
                x = float(valor)
            else:
                x = float(row['Center X'])
                
            if isinstance(row['Center Y'], str) and 'tensor' in row['Center Y']:
                # Extraer el número entre paréntesis
                valor = row['Center Y'].split('(')[1].split(')')[0]
                y = float(valor)
            else:
                y = float(row['Center Y'])
                
            # Convertir a enteros para coordenadas de píxeles
            x = int(x)
            y = int(y)
            
            # Procesar el área de manera similar
            if isinstance(row['Ellipse Area (pixels^2)'], str) and 'tensor' in row['Ellipse Area (pixels^2)']:
                # Extraer el número entre paréntesis
                valor = row['Ellipse Area (pixels^2)'].split('(')[1].split(')')[0]
                area = float(valor)
            else:
                area = float(row['Ellipse Area (pixels^2)'])
                
            # Determinar a qué cuadrante pertenece
            cuad_col = x // cuad_width
            cuad_row = y // cuad_height
            
            # Ajustar para asegurar que está dentro de los límites
            cuad_col = min(2, max(0, cuad_col))
            cuad_row = min(2, max(0, cuad_row))
            
            # Índice del cuadrante (0-8)
            cuad_idx = cuad_row * 3 + cuad_col
            
            # Acumular área en este cuadrante
            areas_por_cuadrante[cuad_idx] += area
            
            # Guardar referencia al canal
            canales_por_cuadrante[cuad_idx].append((x, y, area))
        except Exception as e:
            print(f"Error procesando canal {i}: {e}")
            continue
        
        # Determinar a qué cuadrante pertenece
        cuad_col = x // cuad_width
        cuad_row = y // cuad_height
        
        # Ajustar para asegurar que está dentro de los límites
        cuad_col = min(2, max(0, cuad_col))
        cuad_row = min(2, max(0, cuad_row))
        
        # Índice del cuadrante (0-8)
        cuad_idx = cuad_row * 3 + cuad_col
        
        # Acumular área en este cuadrante
        areas_por_cuadrante[cuad_idx] += area
        
        # Guardar referencia al canal
        canales_por_cuadrante[cuad_idx].append((x, y, area))
    
    # Encontrar cuadrante con mayor área
    cuad_max_area_idx = np.argmax(areas_por_cuadrante)
    
    # Imagen para visualización
    imagen_con_cuadrantes = imagen.copy()
    
    # Dibujar líneas de cuadrantes
    for i in range(1, 3):
        # Líneas horizontales
        cv2.line(imagen_con_cuadrantes, (0, i*cuad_height), 
                 (width, i*cuad_height), (255, 255, 255), 2)
        # Líneas verticales
        cv2.line(imagen_con_cuadrantes, (i*cuad_width, 0), 
                 (i*cuad_width, height), (255, 255, 255), 2)
    
    # Marcar cuadrante con mayor área
    max_row = cuad_max_area_idx // 3
    max_col = cuad_max_area_idx % 3
    x1 = max_col * cuad_width
    y1 = max_row * cuad_height
    x2 = (max_col + 1) * cuad_width
    y2 = (max_row + 1) * cuad_height
    
    # Dibujar rectángulo semitransparente en el cuadrante con mayor área
    overlay = imagen_con_cuadrantes.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, imagen_con_cuadrantes, 0.7, 0, imagen_con_cuadrantes)
    
    # Añadir texto con información por cuadrante
    for i in range(9):
        row = i // 3
        col = i % 3
        text_x = col * cuad_width + 10
        text_y = row * cuad_height + 30
        
        # Texto con área total y número de canales
        text = f"Area: {areas_por_cuadrante[i]:.1f}"
        cv2.putText(imagen_con_cuadrantes, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = f"Canales: {len(canales_por_cuadrante[i])}"
        cv2.putText(imagen_con_cuadrantes, text, (text_x, text_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Guardar imagen final
    cv2.imwrite(output_path, imagen_con_cuadrantes)
    
    return imagen_con_cuadrantes, areas_por_cuadrante, canales_por_cuadrante

def visualizar_resultados_cuadrantes(root, imagen_path, areas_por_cuadrante, canales_por_cuadrante):
    """
    Muestra los resultados del análisis por cuadrantes en una interfaz gráfica.
    
    Args:
        root: Ventana raíz de Tkinter
        imagen_path: Ruta a la imagen con cuadrantes analizados
        areas_por_cuadrante: Array con áreas por cuadrante
        canales_por_cuadrante: Lista de canales agrupados por cuadrante
    """
    for widget in root.winfo_children():
        widget.destroy()
    
    configure_window(root, "Análisis por Cuadrantes")
    
    # Crear pestañas
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)
    
    # Pestaña de visualización
    frame1 = Frame(notebook, bg='#001f3f')
    frame2 = Frame(notebook, bg='#001f3f')
    notebook.add(frame1, text="Visualización")
    notebook.add(frame2, text="Datos por Cuadrante")
    
    # Mostrar imagen
    try:
        img = Image.open(imagen_path)
        img = img.resize((800, 800), Image.LANCZOS)  # Ajustar tamaño para la visualización
        img_tk = ImageTk.PhotoImage(img)
        
        img_label = Label(frame1, image=img_tk, bg='#001f3f')
        img_label.image = img_tk  # Mantener referencia
        img_label.pack(pady=20)
        
        # Botón para guardar imagen
        def guardar_imagen():
            destino = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                initialfile="analisis_cuadrantes.png")
            if destino:
                import shutil
                shutil.copy(imagen_path, destino)
                messagebox.showinfo("Éxito", f"Imagen guardada en {destino}")
                
        save_button = Button(frame1, text="Guardar Imagen", command=guardar_imagen)
        configure_button(save_button)
        save_button.pack(pady=10)
    except Exception as e:
        error_label = Label(frame1, text=f"Error al cargar la imagen: {e}", 
                           fg="white", bg='#001f3f')
        error_label.pack(pady=20)
    
    # Mostrar datos por cuadrante
    text_area = Text(frame2, bg='#001f3f', fg="white", font=("Helvetica", 12))
    text_area.pack(fill='both', expand=True, padx=20, pady=20)
    
    text_area.insert('1.0', "ANÁLISIS POR CUADRANTES\n\n")
    
    # Encontrar cuadrante con mayor área
    cuad_max_area_idx = np.argmax(areas_por_cuadrante)
    
    # Mostrar información por cuadrante
    for i in range(9):
        row = i // 3
        col = i % 3
        
        # Destacar el cuadrante con mayor área
        if i == cuad_max_area_idx:
            text_area.insert('end', f"CUADRANTE {i+1} (MAYOR DENSIDAD) - Fila {row+1}, Columna {col+1}\n")
        else:
            text_area.insert('end', f"CUADRANTE {i+1} - Fila {row+1}, Columna {col+1}\n")
            
        text_area.insert('end', f"  Área total: {areas_por_cuadrante[i]:.2f} pixels²\n")
        text_area.insert('end', f"  Número de canales: {len(canales_por_cuadrante[i])}\n")
        
        if len(canales_por_cuadrante[i]) > 0:
            area_promedio = sum(canal[2] for canal in canales_por_cuadrante[i]) / len(canales_por_cuadrante[i])
            text_area.insert('end', f"  Área promedio por canal: {area_promedio:.2f} pixels²\n")
        
        text_area.insert('end', "\n")
    
    # Botón para exportar resultados a Excel
    def exportar_excel():
        destino = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="resultados_cuadrantes.xlsx")
        
        if destino:
            # Crear DataFrame con resultados
            data = []
            for i in range(9):
                row = i // 3
                col = i % 3
                
                data.append({
                    'Cuadrante': i+1,
                    'Fila': row+1,
                    'Columna': col+1,
                    'Area Total': areas_por_cuadrante[i],
                    'Num Canales': len(canales_por_cuadrante[i]),
                    'Area Promedio': sum(canal[2] for canal in canales_por_cuadrante[i]) / max(1, len(canales_por_cuadrante[i])),
                    'Mayor Densidad': 'Sí' if i == cuad_max_area_idx else 'No'
                })
                
            df = pd.DataFrame(data)
            df.to_excel(destino, index=False)
            messagebox.showinfo("Éxito", f"Datos exportados a {destino}")
    
    export_button = Button(frame2, text="Exportar a Excel", command=exportar_excel)
    configure_button(export_button)
    export_button.pack(pady=10)

def main():
    """Función principal del programa"""
    # Crear ventana principal
    root = Tk()
    
    # Título de la ventana
    configure_window(root, "Análisis de Canales por Cuadrantes")
    
    # Función para iniciar el análisis
    def iniciar_analisis():
        # Seleccionar imagen original
        imagen_original = filedialog.askopenfilename(
            title="Seleccione la imagen original",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        
        if not imagen_original:
            return
            
        # Seleccionar archivo Excel con detecciones
        excel_path = filedialog.askopenfilename(
            title="Seleccione el archivo Excel con las detecciones",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if not excel_path:
            return
            
        # Mostrar ventana de progreso
        progreso = Toplevel(root)
        configure_window(progreso, "Procesando")
        Label(progreso, text="Procesando imagen...", font=("Helvetica", 16),
              fg="white", bg='#001f3f').pack(expand=True)
        progreso.update()
        
        try:
            # Reconstruir imagen con detecciones
            imagen_reconstruida_path = "imagen_reconstruida.png"
            imagen = reconstruir_imagen_con_detecciones(
                imagen_original, excel_path, imagen_reconstruida_path
            )
            
            # Cargar datos de detecciones
            df = pd.read_excel(excel_path)
            
            # Analizar por cuadrantes
            imagen_cuadrantes_path = "imagen_cuadrantes.png"
            imagen_final, areas, canales = analizar_cuadrantes(
                imagen, df, imagen_cuadrantes_path
            )
            
            # Cerrar ventana de progreso
            progreso.destroy()
            
            # Mostrar resultados
            visualizar_resultados_cuadrantes(root, imagen_cuadrantes_path, areas, canales)
            
        except Exception as e:
            progreso.destroy()
            messagebox.showerror("Error", f"Error durante el análisis: {e}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Crear botones en la ventana principal
    titulo = Label(root, text="Análisis de Canales de Havers por Cuadrantes", 
                  font=("Helvetica", 24), fg="white", bg='#001f3f')
    titulo.pack(pady=30)
    
    descripcion = Label(root, text="Esta aplicación analiza la distribución de canales de Havers por cuadrantes\n"
                       "y destaca el cuadrante con mayor densidad de canales.",
                       font=("Helvetica", 14), fg="white", bg='#001f3f')
    descripcion.pack(pady=20)
    
    iniciar_button = Button(root, text="Iniciar Análisis", command=iniciar_analisis)
    configure_button(iniciar_button)
    iniciar_button.pack(pady=30)
    
    # Iniciar la aplicación
    root.mainloop()

if __name__ == '__main__':
    main()