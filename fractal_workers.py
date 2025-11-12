"""
Archivo de Worker para Fractal Zoom v5.0
Contiene la(s) funcion(es) que serán ejecutadas por
el pool de multiprocessing.
"""

import numpy as np
import random

def find_best_square_histogram(target_idx, patches, N, ks, metric, num_candidates):
    """
    Worker de matching v5.0.
    Usa los histogramas pre-calculados. No usa cv2.resize/ssim.
    Esta función VIVE en un archivo separado para ser compatible con 'spawn'.
    """
    target = patches[target_idx]
    hist_target_raw = target['hist_vec_raw']
    
    # Normalizar el vector del objetivo
    sum_target = np.sum(hist_target_raw)
    if sum_target == 0:
        # Es un parche vacío/negro, no se puede comparar
        return {'top': target['id'], 'k':1, 'score': 0.0}
    
    hist_target_norm = hist_target_raw.astype(np.float32) / sum_target
    norm_target = np.linalg.norm(hist_target_norm) # Pre-calcular para Coseno
    
    best = None
    # Coseno: queremos max (cercano a 1)
    # Euclidiana: queremos min (cercano a 0)
    best_score = float('-inf') if "cosine" in metric else float('inf')

    for k in ks:
        # Generar candidatos aleatorios
        candidates = []
        for _ in range(num_candidates):
            rt = random.randint(0, N - k)
            ct = random.randint(0, N - k)
            candidates.append((rt, ct))

        for (rt, ct) in candidates:
            
            # --- INGENUIDAD (TU LÓGICA) ---
            # 1. Sumar los vectores del bloque k*k
            # (Inicializar con el tipo de dato del vector crudo)
            hist_block_raw = np.zeros_like(hist_target_raw)
            for rr in range(k):
                for cc in range(k):
                    idx = (rt + rr) * N + (ct + cc)
                    hist_block_raw += patches[idx]['hist_vec_raw']
            
            # 2. Normalizar el vector del bloque
            sum_block = np.sum(hist_block_raw)
            if sum_block == 0:
                continue # No se puede comparar un bloque vacío
            
            hist_block_norm = hist_block_raw.astype(np.float32) / sum_block
            
            # --- COMPARACIÓN DE VECTORES (muy rápido) ---
            score = 0
            better = False
            
            if metric == 'histogram_cosine':
                # Similitud Coseno
                norm_block = np.linalg.norm(hist_block_norm)
                if norm_target == 0 or norm_block == 0:
                    score = 0.0
                else:
                    score = np.dot(hist_target_norm, hist_block_norm) / (norm_target * norm_block)
                
                better = (score > best_score + 1e-9) or \
                         (abs(score-best_score) < 1e-9 and (best and k > best['k']))
            
            elif metric == 'histogram_euclidean':
                # Distancia Euclidiana
                score = np.linalg.norm(hist_target_norm - hist_block_norm)
                better = (score < best_score - 1e-6) or \
                         (abs(score-best_score) < 1e-6 and (best and k > best['k']))
            
            if better:
                best_score = score
                best = {'top':(rt, ct), 'k':k, 'score':float(score)}
    
    if best is None: # Fallback
        best = {'top': target['id'], 'k':1, 'score': 0.0}
    return best