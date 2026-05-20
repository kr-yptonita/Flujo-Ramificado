import subprocess
import os

def crear_animacion(carpeta_origen, archivo_salida, fps=24):
    print(f"Buscando imágenes en {carpeta_origen}...")
    
    # Verificar si la carpeta existe
    if not os.path.exists(carpeta_origen):
        print(f"Error: La carpeta '{carpeta_origen}' no existe.")
        return
        
    print(f"Generando video a {fps} fps usando ffmpeg...")
    
    # Comando de ffmpeg
    # -y : sobreescribir archivo de salida sin preguntar
    # -framerate fps : tasa de fotogramas por segundo
    # -i .../frame_%04d.png : patrón de entrada de los archivos
    # -c:v libx264 : codec de video H.264
    # -pix_fmt yuv420p : formato de píxeles para máxima compatibilidad
    comando = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(carpeta_origen, "frame_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        archivo_salida
    ]
    
    try:
        subprocess.run(comando, check=True, capture_output=True, text=True)
        print(f"¡Animación guardada exitosamente como '{archivo_salida}'!")
    except subprocess.CalledProcessError as e:
        print("Error al generar la animación. Salida de ffmpeg:")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: No se encontró 'ffmpeg'. Por favor, asegúrate de tenerlo instalado en tu sistema.")

if __name__ == "__main__":
    # Ruta relativa a la carpeta de los fotogramas
    carpeta = "Trayectorias_Fotones_Estocastico"
    # Nombre del archivo de salida
    salida = "animacion_trayectorias_estocasticas.mp4"
    
    # Llamar a la función con 24 fotogramas por segundo
    crear_animacion(carpeta, salida, fps=24)
