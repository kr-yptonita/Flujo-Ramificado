import numpy as np
import time
import os
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class TrazadorFotones:
    def __init__(self, matriz_espesores):
        self.T = matriz_espesores.astype(np.float32)
        self.H_pix, self.W_pix = self.T.shape
        self.H_nodes = self.H_pix + 1
        self.W_nodes = self.W_pix + 1
        
        self.dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        self.dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        self.L = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)])
        
        self.N = self.H_nodes * self.W_nodes * 8
        
        print("    Construyendo grafo sparse...")
        start = time.time()
        
        # Optimizacion Vectorizada masiva: de 70 segundos a milisegundos
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
            final_peso = peso[valid_weights] + epsilon
            
            end_nodes = (ny_v[valid_weights] * self.W_nodes + nx_v[valid_weights]) * 8 + d_vec_v[valid_weights]
            start_n = start_nodes[valid][valid_weights]
            
            all_rows.append(start_n)
            all_cols.append(end_nodes)
            all_data.append(final_peso)
            
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.concatenate(all_data)
        
        self.graph = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        print(f"    Grafo construido en {time.time() - start:.2f} s")

    def trazar_camino(self, x_ini, y_ini):
        # Determinamos automáticamente los caminos iniciales, meta y bordes letales
        if x_ini == 0:
            caminos_iniciales = [
                (1, y_ini - 1, 7), (1, y_ini, 0), (1, y_ini + 1, 1)  
            ]
            target_edge = 'right'
            lethal_edges = ['top', 'bottom']
        elif x_ini == self.W_nodes - 1:
            caminos_iniciales = [
                (x_ini - 1, y_ini + 1, 3), (x_ini - 1, y_ini, 4), (x_ini - 1, y_ini - 1, 5)  
            ]
            target_edge = 'left'
            lethal_edges = ['top', 'bottom']
        elif y_ini == 0:
            caminos_iniciales = [
                (x_ini + 1, 1, 1), (x_ini, 1, 2), (x_ini - 1, 1, 3)  
            ]
            target_edge = 'bottom'
            lethal_edges = ['left', 'right']
        elif y_ini == self.H_nodes - 1:
            caminos_iniciales = [
                (x_ini - 1, y_ini - 1, 5), (x_ini, y_ini - 1, 6), (x_ini + 1, y_ini - 1, 7)  
            ]
            target_edge = 'top'
            lethal_edges = ['left', 'right']
        else:
            print("El punto inicial no se encuentra en un borde válido.")
            return []
            
        mejor_ruta = []
        mejor_costo = np.inf
        
        for nx, ny, ndir in caminos_iniciales:
            if 0 <= ny < self.H_nodes and 0 <= nx < self.W_nodes:
                ruta, costo = self._trazar_camino_interno(nx, ny, ndir, target_edge, lethal_edges, x_ini, y_ini)
                if costo < mejor_costo and len(ruta) > 0:
                    mejor_costo = costo
                    mejor_ruta = [(x_ini, y_ini)] + ruta
                    
        return mejor_ruta

    def _trazar_camino_interno(self, x_ini, y_ini, dir_ini, target_edge, lethal_edges, orig_x, orig_y):
        start_idx = (y_ini * self.W_nodes + x_ini) * 8 + dir_ini
        
        dist_matrix, predecessors = shortest_path(csgraph=self.graph, directed=True, indices=start_idx, return_predecessors=True)
        
        best_cost = np.inf
        best_end_idx = -1
        
        target_nodes = []
        if target_edge == 'right':
            x = self.W_nodes - 1
            for y in range(self.H_nodes):
                for d in range(8):
                    target_nodes.append((y * self.W_nodes + x) * 8 + d)
        elif target_edge == 'left':
            x = 0
            for y in range(self.H_nodes):
                for d in range(8):
                    target_nodes.append((y * self.W_nodes + x) * 8 + d)
        elif target_edge == 'bottom':
            y = self.H_nodes - 1
            for x in range(self.W_nodes):
                for d in range(8):
                    target_nodes.append((y * self.W_nodes + x) * 8 + d)
        elif target_edge == 'top':
            y = 0
            for x in range(self.W_nodes):
                for d in range(8):
                    target_nodes.append((y * self.W_nodes + x) * 8 + d)
                    
        for idx in target_nodes:
            if dist_matrix[idx] < best_cost:
                best_cost = dist_matrix[idx]
                best_end_idx = idx
                    
        if best_end_idx == -1 or best_cost == np.inf:
            return [], np.inf
            
        ruta = []
        curr = best_end_idx
        while curr != -9999 and curr >= 0:
            d = curr % 8
            curr_no_d = curr // 8
            x = curr_no_d % self.W_nodes
            y = curr_no_d // self.W_nodes
            ruta.append((x, y))
            if curr == start_idx:
                break
            curr = predecessors[curr]
            
        ruta.reverse()
        
        ruta_truncada = []
        for px, py in ruta:
            ruta_truncada.append((px, py))
            if px == orig_x and py == orig_y:
                continue
                
            hit_lethal = False
            for edge in lethal_edges:
                if edge == 'top' and py == 0: hit_lethal = True
                elif edge == 'bottom' and py == self.H_nodes - 1: hit_lethal = True
                elif edge == 'left' and px == 0: hit_lethal = True
                elif edge == 'right' and px == self.W_nodes - 1: hit_lethal = True
                
            if hit_lethal:
                break
                
        return ruta_truncada, best_cost

if __name__ == "__main__":
    directorio_base = os.path.dirname(os.path.abspath(__file__))
    archivo_npy = os.path.join(directorio_base, "Matrices_Espesor", "frame00086400.npy")
    
    if os.path.exists(archivo_npy):
        print("Cargando matriz...")
        T = np.load(archivo_npy)
        
        trazador = TrazadorFotones(T)
        
        print("Calculando rutas del haz de luz...")
        start = time.time()
        
        x_ini = 0
        y_center = T.shape[0] // 2
        rutas = []
        for i in range(4):
            ruta = trazador.trazar_camino(x_ini, y_center + i, dir_ini=0)
            if ruta:
                rutas.append(ruta)
                
        end = time.time()
        print(f"{len(rutas)} Rutas calculadas en {end - start:.2f} s en total.")
        
        if rutas:
            plt.figure(figsize=(12, 8))
            vmax = np.percentile(T, 99)
            vmin = np.percentile(T, 1)
            plt.imshow(T, cmap='inferno', vmin=vmin, vmax=vmax, extent=[0, T.shape[1], T.shape[0], 0])
            
            for ruta in rutas:
                rx = [p[0] for p in ruta]
                ry = [p[1] for p in ruta]
                plt.plot(rx, ry, color='#00FF00', linewidth=1)
            
            plt.colorbar(label='Espesor (nm)')
            plt.title('Trayectoria Óptima del Fotón')
            plt.axis('off')
            
            salida = os.path.join(directorio_base, "heatmap_foton.png")
            plt.savefig(salida, dpi=150, bbox_inches='tight', pad_inches=0)
            print("Heatmap con haz guardado en", salida)
        else:
            print("No se encontró una ruta válida.")
    else:
        print(f"No se encontró el archivo: {archivo_npy}")
