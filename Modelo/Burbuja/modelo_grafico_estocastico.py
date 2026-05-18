import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import importlib
import time
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Añadir directorio actual al path para importaciones
directorio_base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directorio_base)

# Importación dinámica por espacio en el nombre
gen_matrices = importlib.import_module("Generador de matrices")
from Trazador_de_fotones import TrazadorFotones

class TrazadorFotonesEstocastico(TrazadorFotones):
    def __init__(self, matriz_espesores, prob_mantener_carril=0.85, costo_penalizacion=1000.0):
        self.T = matriz_espesores.astype(np.float32)
        self.H_pix, self.W_pix = self.T.shape
        self.H_nodes = self.H_pix + 1
        self.W_nodes = self.W_pix + 1
        
        self.dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        self.dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        self.L = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])
        
        self.N = self.H_nodes * self.W_nodes * 8
        
        print("    Construyendo grafo sparse estocástico...")
        start = time.time()
        
        # Optimizacion Vectorizada masiva
        T_pad = np.pad(self.T, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
        
        y, x, d = np.meshgrid(np.arange(self.H_nodes), np.arange(self.W_nodes), np.arange(8), indexing='ij')
        y = y.flatten()
        x = x.flatten()
        d = d.flatten()
        
        start_nodes = (y * self.W_nodes + x) * 8 + d
        
        all_rows = []
        all_cols = []
        all_data = []
        
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        
        for offset in [-1, 0, 1]:
            d_vecino = (d + offset) % 8
            nx = x + self.dx[d_vecino]
            ny = y + self.dy[d_vecino]
            
            valid = (nx >= 0) & (nx < self.W_nodes) & (ny >= 0) & (ny < self.H_nodes)
            
            y_v = y[valid]
            x_v = x[valid]
            d_v = d[valid]
            nx_v = nx[valid]
            ny_v = ny[valid]
            d_vec_v = d_vecino[valid]
            
            peso = np.full(len(y_v), np.nan, dtype=np.float32)
            
            mask0 = (d_vec_v == 0)
            if np.any(mask0):
                p1 = T_pad[y_v[mask0], x_v[mask0]+1]
                p2 = T_pad[y_v[mask0]+1, x_v[mask0]+1]
                peso[mask0] = np.nanmean([p1, p2], axis=0) * self.L[0]
                
            mask1 = (d_vec_v == 1)
            if np.any(mask1):
                peso[mask1] = T_pad[y_v[mask1]+1, x_v[mask1]+1] * self.L[1]
                
            mask2 = (d_vec_v == 2)
            if np.any(mask2):
                p1 = T_pad[y_v[mask2]+1, x_v[mask2]]
                p2 = T_pad[y_v[mask2]+1, x_v[mask2]+1]
                peso[mask2] = np.nanmean([p1, p2], axis=0) * self.L[2]
                
            mask3 = (d_vec_v == 3)
            if np.any(mask3):
                peso[mask3] = T_pad[y_v[mask3]+1, x_v[mask3]] * self.L[3]
                
            mask4 = (d_vec_v == 4)
            if np.any(mask4):
                p1 = T_pad[y_v[mask4], x_v[mask4]]
                p2 = T_pad[y_v[mask4]+1, x_v[mask4]]
                peso[mask4] = np.nanmean([p1, p2], axis=0) * self.L[4]
                
            mask5 = (d_vec_v == 5)
            if np.any(mask5):
                peso[mask5] = T_pad[y_v[mask5], x_v[mask5]] * self.L[5]
                
            mask6 = (d_vec_v == 6)
            if np.any(mask6):
                p1 = T_pad[y_v[mask6], x_v[mask6]]
                p2 = T_pad[y_v[mask6], x_v[mask6]+1]
                peso[mask6] = np.nanmean([p1, p2], axis=0) * self.L[6]
                
            mask7 = (d_vec_v == 7)
            if np.any(mask7):
                peso[mask7] = T_pad[y_v[mask7], x_v[mask7]+1] * self.L[7]
                
            valid_weights = ~np.isnan(peso)
            
            cambio_dir = np.abs((d_vec_v[valid_weights] - d_v[valid_weights] + 4) % 8 - 4)
            epsilon = cambio_dir * 1e-9
            
            # --- PROCESO ESTOCÁSTICO ---
            # Si hay un cambio de dirección (cambio_dir != 0), agregamos una probabilidad
            # independiente de penalizar fuertemente el giro para forzar al fotón a 
            # mantener su carril.
            penalizacion_estocastica = np.where(
                np.random.rand(len(valid_weights)) < prob_mantener_carril,
                costo_penalizacion * cambio_dir,
                0.0
            )
            
            final_peso = peso[valid_weights] + epsilon + penalizacion_estocastica
            # -----------------------------
            
            end_nodes = (ny_v[valid_weights] * self.W_nodes + nx_v[valid_weights]) * 8 + d_vec_v[valid_weights]
            start_n = start_nodes[valid][valid_weights]
            
            all_rows.append(start_n)
            all_cols.append(end_nodes)
            all_data.append(final_peso)
            
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)
        
        self.graph = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        print(f"    Grafo estocástico construido en {time.time() - start:.2f} s")

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
    
    # Cambiamos nombre de carpetas y archivos para no sobreescribir el original
    carpeta_trayectorias = os.path.join(directorio_base, "Trayectorias_Estocasticas")
    video_salida = os.path.join(directorio_base, "animacion_burbuja_estocastica.mp4")
    
    if not os.path.exists(carpeta_trayectorias):
        os.makedirs(carpeta_trayectorias)
        
    print("Generando Tabla de Colores (LUT)...")
    espesores, lut_bgr = gen_matrices.crear_lut_espesores()
    
    # Limitado a 24 fotogramas para prueba rápida como en el original
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
        
    print(f"Iniciando trazado de fotones estocástico para {len(rutas_npy)} frames...")
    
    for i, archivo_npy in enumerate(rutas_npy):
        print(f"\nProcesando Frame {i+1}/{len(rutas_npy)}: {os.path.basename(archivo_npy)}")
        
        T = np.load(archivo_npy)
        # USAR EL NUEVO TRAZADOR ESTOCÁSTICO
        trazador = TrazadorFotonesEstocastico(T, prob_mantener_carril=0.85, costo_penalizacion=1000.0)
        
        y_ini = 0 # Borde superior
        x_center = T.shape[1] // 2
        
        # Haz de 4 fotones
        rutas_haz = []
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
        plt.title(f'Trayectoria Estocástica del Fotón - Frame {i+1:04d}')
        plt.axis('off')
        
        nombre_salida = f"frame_{i+1:04d}.png"
        ruta_salida = os.path.join(carpeta_trayectorias, nombre_salida)
        
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Liberar memoria RAM crucial
        print(f"  Imagen guardada: {nombre_salida}")
        
    print("\nIniciando compilación de video...")
    compilar_video(carpeta_trayectorias, video_salida, fps=24)

if __name__ == '__main__':
    main()
