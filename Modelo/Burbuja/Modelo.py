import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Añadir directorio actual al path para importaciones
directorio_base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directorio_base)

# Importación dinámica por espacio en el nombre
gen_matrices = importlib.import_module("Generador de matrices")
from Trazador_de_fotones import TrazadorFotones

def compilar_video(carpeta_imagenes, salida_video, fps=24):
    import glob
    imagenes = sorted(glob.glob(os.path.join(carpeta_imagenes, "*.png")))
    if not imagenes:
        print("No hay imágenes para el video.")
        return
        
    frame = cv2.imread(imagenes[0])
    h, w, layers = frame.shape
    size = (w, h)
    
    # Usar codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(salida_video, fourcc, fps, size)
    
    print(f"Compilando {len(imagenes)} frames en {salida_video} a {fps} fps...")
    for i, archivo in enumerate(imagenes):
        img = cv2.imread(archivo)
        # Redimensionar por si el bbox_inches='tight' varió el tamaño en 1 o 2 píxeles
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        out.write(img)
        if (i+1) % 5 == 0:
            print(f"Frame {i+1}/{len(imagenes)} agregado al video.")
            
    out.release()
    print("Video compilado exitosamente.")

def main():
    carpeta_frames = os.path.join(directorio_base, "Frames")
    carpeta_salida_matrices = os.path.join(directorio_base, "Matrices_Espesor")
    carpeta_trayectorias = os.path.join(directorio_base, "Trayectorias")
    video_salida = os.path.join(directorio_base, "animacion_burbuja.mp4")
    
    if not os.path.exists(carpeta_trayectorias):
        os.makedirs(carpeta_trayectorias)
        
    print("Generando Tabla de Colores (LUT)...")
    espesores, lut_bgr = gen_matrices.crear_lut_espesores()
    
    # PROCESAMIENTO DE PRUEBA: limitamos a 24 fotogramas para no demorar 65 horas.
    limite = 24
    print(f"Generando matrices (limitado a {limite} frames para prueba)...")
    rutas_npy = gen_matrices.procesar_fotogramas(
        carpeta_frames, 
        carpeta_salida_matrices, 
        espesores, 
        lut_bgr, 
        limite_frames=limite
    )
    
    if not rutas_npy:
        print("No se generaron matrices.")
        return
        
    print(f"Iniciando trazado de fotones para {len(rutas_npy)} frames...")
    
    for i, archivo_npy in enumerate(rutas_npy):
        print(f"\nProcesando Frame {i+1}/{len(rutas_npy)}: {os.path.basename(archivo_npy)}")
        
        T = np.load(archivo_npy)
        trazador = TrazadorFotones(T)
        
        y_ini = 0 # Borde superior
        x_center = T.shape[1] // 2
        
        # Haz de 4 fotones
        rutas_haz = []
        import time
        start_trazado = time.time()
        
        for p in range(4):
            x_ini = x_center + p * 10
            print(f"  Trazando fotón {p+1} (origen x={x_ini}, y={y_ini})...")
            ruta = trazador.trazar_camino(x_ini, y_ini)
            if ruta:
                rutas_haz.append(ruta)
                
        end_trazado = time.time()
        print(f"  Trazado completado en {end_trazado - start_trazado:.2f} s")
        
        # Guardar imagen con trayectorias
        fig = plt.figure(figsize=(12, 8))
        vmax = np.percentile(T, 99)
        vmin = np.percentile(T, 1)
        plt.imshow(T, cmap='inferno', vmin=vmin, vmax=vmax, extent=[0, T.shape[1], T.shape[0], 0])
        
        for ruta in rutas_haz:
            rx = [p[0] for p in ruta]
            ry = [p[1] for p in ruta]
            plt.plot(rx, ry, color='#00FF00', linewidth=1) # 1 píxel de grosor
            
        plt.colorbar(label='Espesor (nm)')
        plt.title(f'Trayectoria Óptima del Fotón - Frame {i+1:04d}')
        plt.axis('off')
        
        nombre_salida = f"frame_{i+1:04d}.png"
        ruta_salida = os.path.join(carpeta_trayectorias, nombre_salida)
        
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Liberar memoria RAM crucial para evitar fugas en el bucle
        print(f"  Imagen guardada: {nombre_salida}")
        
    print("\nIniciando compilación de video...")
    compilar_video(carpeta_trayectorias, video_salida, fps=24)

if __name__ == '__main__':
    main()
