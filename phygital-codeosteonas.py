import cv2
import os
import shutil
import pandas as pd
from math import ceil, pi
from ultralytics import YOLO
import torch
from tkinter import Tk, Button, Text, Scrollbar, Canvas, Frame, Label, ttk, filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import io
import numpy as np

def configure_window(window, title):
    window.title(title)
    window.geometry("1000x1000")
    window.configure(bg='#001f3f')
def select_image(root):
    def on_button_click():
        file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            global image_path
            image_path = file_path
            root.quit()
            for widget in root.winfo_children():
                widget.destroy()

    configure_window(root, "Select Image")
    title = Label(root, text="PhigytalBone 2.0", font=("Helvetica", 24), fg="white", bg="#001f3f")
    title.pack(pady=20)

    button = Button(root, text="Load Image", command=on_button_click)
    configure_button(button)
    button.pack(expand=True)

    root.mainloop()
    return image_path

def divide_and_save_image(image_path, output_dir, num_segments=150):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image {image_path}")
        return None, None, None

    cols = 15
    rows = ceil(num_segments / cols)
    segment_height = image.shape[0] // rows
    segment_width = image.shape[1] // cols

    segment_positions = []
    for i in range(rows):
        for j in range(cols):
            start_y = i * segment_height
            end_y = start_y + segment_height if i < rows - 1 else image.shape[0]
            start_x = j * segment_width
            end_x = start_x + segment_width if j < cols - 1 else image.shape[1]
            segment = image[start_y:end_y, start_x:end_x]
            segment_filename = f"segment_{i * cols + j + 1}.png"
            cv2.imwrite(os.path.join(output_dir, segment_filename), segment)
            segment_positions.append((start_x, start_y, i * cols + j + 1))

    return segment_positions, segment_width, segment_height

def calculate_box_centers_and_areas(boxes, start_x, start_y, segment_id):
    centers = []
    for box in boxes:
        xyxy = box.xyxy.clone().detach().cpu().view(1, 4)
        width = xyxy[0, 2] - xyxy[0, 0]
        height = xyxy[0, 3] - xyxy[0, 1]
        cx = start_x + (xyxy[0, 0] + xyxy[0, 2]) / 2
        cy = start_y + (xyxy[0, 1] + xyxy[0, 3]) / 2
        semi_major_axis = width / 2
        semi_minor_axis = height / 2
        ellipse_area = pi * semi_major_axis * semi_minor_axis  # Calcula el área de la elipse
        centers.append((cx, cy, segment_id, ellipse_area))
    return centers

def plot_centers(df, image_path):
    plt.figure(figsize=(10, 10))
    # Cargar la imagen inicial
    image = plt.imread(image_path)
    plt.imshow(image, extent=[0, image.shape[1], image.shape[0], 0], alpha=0.6)
    plt.scatter(df['Center X'], df['Center Y'], c='blue', marker='o', s=10)  # Puntos más pequeños
    plt.title('Mapa de coordenadas de canales de Havers')
    plt.xlabel('Center X')
    plt.ylabel('Center Y')
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def plot_heatmap(df, image_path):
    plt.figure(figsize=(10, 10))
    # Generar el mapa de calor
    heatmap, xedges, yedges = np.histogram2d(df['Center X'], df['Center Y'], bins=(100, 100))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=0.6)
    plt.colorbar()
    # Cargar la imagen inicial con transparencia
    image = plt.imread(image_path)
    plt.imshow(image, extent=[0, image.shape[1], image.shape[0], 0], alpha=0.4)
    plt.title('Mapa de densidad de canales de Havers')
    plt.xlabel('Center X')
    plt.ylabel('Center Y')
    plt.grid(True)
    plt.gca().invert_yaxis()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def save_plot(buf, title):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title=title
    )
    if file_path:
        with open(file_path, 'wb') as f:
            f.write(buf.getbuffer())

def display_results(root, excel_path, plot_buf, heatmap_buf, avg_area, count_havers):
    for widget in root.winfo_children():
        widget.destroy()
    configure_window(root, "Results")
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Crear pestañas
    frame1 = Frame(notebook, bg='#001f3f')
    frame2 = Frame(notebook, bg='#001f3f')
    frame3 = Frame(notebook, bg='#001f3f')
    frame4 = Frame(notebook, bg='#001f3f')
    notebook.add(frame1, text="Coordinates")
    notebook.add(frame2, text="Plot")
    notebook.add(frame3, text="Analysis")
    notebook.add(frame4, text="Heatmap")

    # Pestaña de coordenadas
    text_area = Text(frame1, bg='#001f3f', fg="white", insertbackground="white")
    text_area.pack(side='left', fill='both', expand=True)
    scrollbar = Scrollbar(frame1, command=text_area.yview)
    scrollbar.pack(side='right', fill='y')
    text_area.config(yscrollcommand=scrollbar.set)


    # Mostrar contenido del archivo de Excel
    with pd.ExcelFile(excel_path) as xls:
        df = pd.read_excel(xls)
        text_area.insert('1.0', df.to_string())

    # Mostrar el gráfico de centros
    plot_image = Image.open(plot_buf)
    plot_photo = ImageTk.PhotoImage(plot_image)

    # Ventana principal
    root.mainloop()

# Actualizar la ventana para mostrar el progreso
def main():
    root = Tk()
    configure_window(root, "Processing")
    Label(root, text="Processing, please wait...", font=("Helvetica", 16), fg="white", bg='#001f3f').pack(expand=True)
    root.update()

    images_dir = 'imagenes_sep'
    output_dir = 'resultado_img'
    confidence_threshold = 0.4

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO("runs/detect/train13/weights/best.pt")
    images = sorted(
        [f for f in os.listdir(images_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    # Generar DataFrame con coordenadas y áreas
    df = pd.DataFrame(box_centers_and_areas, columns=['Center X', 'Center Y', 'Segment ID', 'Ellipse Area (pixels^2)'])
    excel_path = 'bounding_box_centers.xlsx'
    df.to_excel(excel_path, index=False)
    print("Bounding box centers and areas saved to Excel.")

    # Generar visualizaciones
    plot_buf = plot_centers(df, image_path)
    heatmap_buf = plot_heatmap(df, image_path)
    avg_area = df['Ellipse Area (pixels^2)'].mean()
    count_havers = df.shape[0]

    # Mostrar resultados
    display_results(root, excel_path, plot_buf, heatmap_buf, avg_area, count_havers)

if __name__ == '__main__':
    main()