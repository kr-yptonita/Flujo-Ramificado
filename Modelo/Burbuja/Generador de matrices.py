import cv2
import numpy as np
import os
import glob
import time
from scipy.spatial import cKDTree

def calcular_color_interferencia(t_nm, n=1.42, theta_i_deg=30.0):
    """
    Calcula el color RGB aproximado para un espesor t_nm dado de la película de jabón.
    """
    theta_i = np.radians(theta_i_deg)
    sin_theta_t = np.sin(theta_i) / n
    theta_t = np.arcsin(sin_theta_t)
    
    cos_theta_t = np.cos(theta_t)
    opd = 2 * n * t_nm * cos_theta_t
    
    lambdas = np.linspace(380, 750, 100)
    fase = np.pi * opd / lambdas
    intensidad = np.sin(fase)**2
    
    # Aproximación simple de funciones de coincidencia de color CIE 1931 (Wyman et al.)
    x_bar = 1.065 * np.exp(-0.5 * ((lambdas - 595.8) / 33.33)**2) + 0.366 * np.exp(-0.5 * ((lambdas - 446.8) / 19.44)**2)
    y_bar = 1.014 * np.exp(-0.5 * ((lambdas - 556.3) / 42.63)**2)
    z_bar = 1.839 * np.exp(-0.5 * ((lambdas - 449.8) / 29.0)**2)
    
    X = np.trapezoid(intensidad * x_bar, lambdas)
    Y = np.trapezoid(intensidad * y_bar, lambdas)
    Z = np.trapezoid(intensidad * z_bar, lambdas)
    
    r_lin =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b_lin =  0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    r_lin = np.clip(r_lin, 0, None)
    g_lin = np.clip(g_lin, 0, None)
    b_lin = np.clip(b_lin, 0, None)
    
    return np.array([b_lin, g_lin, r_lin]) # BGR para OpenCV

def crear_lut_espesores(t_min=10, t_max=1500, paso=1, n=1.42, theta_i=30):
    """
    Crea una tabla de búsqueda (LUT) que mapea el espesor al color RGB (BGR).
    """
    espesores = np.arange(t_min, t_max + paso, paso)
    lut_bgr_lin = np.zeros((len(espesores), 3), dtype=np.float32)
    
    for i, t in enumerate(espesores):
        lut_bgr_lin[i] = calcular_color_interferencia(t, n, theta_i)
        
    # Normalizar globalmente
    max_val = np.max(lut_bgr_lin)
    if max_val > 0:
        lut_bgr_lin = lut_bgr_lin / max_val
        
    # Gamma correction sRGB
    lut_bgr_gamma = np.where(lut_bgr_lin <= 0.0031308, 
                             12.92 * lut_bgr_lin, 
                             1.055 * np.power(lut_bgr_lin, 1/2.4) - 0.055)
                             
    lut_bgr = np.clip(lut_bgr_gamma * 255.0, 0, 255).astype(np.uint8)
    return espesores, lut_bgr

def procesar_fotogramas(carpeta_frames, carpeta_salida, lut_espesores, lut_bgr, limite_frames=None):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        
    patron = os.path.join(carpeta_frames, "frame*.png")
    archivos = sorted(glob.glob(patron))
    
    if not archivos:
        print("No se encontraron fotogramas en", carpeta_frames)
        return []
        
    print(f"Se encontraron {len(archivos)} fotogramas. Construyendo árbol KD...")
    tree = cKDTree(lut_bgr)
    
    if limite_frames is not None:
        archivos = archivos[:limite_frames]
        print(f"Modo LIMITADO: Procesando solo {len(archivos)} fotogramas")
    else:
        print(f"Procesando {len(archivos)} fotogramas...")
        
    archivos_salida = []
    for archivo in archivos:
        inicio = time.time()
        nombre = os.path.basename(archivo)
        
        # 1. Leer imagen
        img = cv2.imread(archivo)
        if img is None:
            print(f"Error al leer {archivo}")
            continue
            
        # 2. Promediar subcuadrículas de 2x2 píxeles
        h, w = img.shape[:2]
        nuevo_w = w // 2
        nuevo_h = h // 2
        img_reducida = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
        
        # 3. Mapeo Color -> Espesor
        pixeles = img_reducida.reshape(-1, 3)
        distancias, indices = tree.query(pixeles)
        
        espesores_pixeles = lut_espesores[indices]
        matriz_espesores = espesores_pixeles.reshape((nuevo_h, nuevo_w))
        
        # 4. Guardar resultado
        nombre_salida = nombre.replace('.png', '.npy')
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        np.save(ruta_salida, matriz_espesores)
        archivos_salida.append(ruta_salida)
        
        fin = time.time()
        print(f"Procesado {nombre} -> {nuevo_w}x{nuevo_h} en {fin - inicio:.2f} s")
        
    return archivos_salida

if __name__ == '__main__':
    print("Generando Tabla de Colores (LUT)...")
    espesores, lut_bgr = crear_lut_espesores(t_min=10, t_max=1500, paso=1, n=1.42, theta_i=30)
    print(f"LUT generada. Rango de espesores: {espesores[0]} - {espesores[-1]} nm")
    
    # Rutas dinámicas basadas en la ubicación del script
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    carpeta_frames = os.path.join(directorio_base, "Frames")
    carpeta_salida = os.path.join(directorio_base, "Matrices_Espesor")
    
    procesar_fotogramas(carpeta_frames, carpeta_salida, espesores, lut_bgr, prueba=True)
