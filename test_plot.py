import numpy as np
import matplotlib.pyplot as plt
import os

directorio_base = os.path.dirname(os.path.abspath(__file__))
archivo_npy = os.path.join(directorio_base, "Modelo/Matrices_Espesor/frame00086400.npy")

espesores = np.load(archivo_npy)

plt.figure(figsize=(12, 8))
# Vemos los primeros espesores (cortamos valores atípicos si los hay para visualizar mejor)
vmax = np.percentile(espesores, 99)
vmin = np.percentile(espesores, 1)

plt.imshow(espesores, cmap='inferno', vmin=vmin, vmax=vmax)
plt.colorbar(label='Espesor (nm)')
plt.title('Mapa de Espesor de Película de Jabón - frame00086400')
plt.axis('off')

salida = os.path.join(directorio_base, "heatmap.png")
plt.savefig(salida, dpi=150, bbox_inches='tight')
print("Heatmap guardado en", salida)
