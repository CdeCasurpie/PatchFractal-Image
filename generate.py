"""
generate.py - Procesamiento de imágenes fractales
Genera patches, calcula matches y guarda todo en /output/<nombre>/
"""

import cv2
import numpy as np
import json
import os
import time
import multiprocessing
from functools import partial
from datetime import datetime
from PIL import Image

# --- CONFIGURACIÓN ---
IMG_SIDE = 1024
INIT_N = 25
MAX_K = 2
RESIZE_C_SIDE = 128
NUM_RANDOM_CANDIDATES = 1000

def load_image_square(path, side):
    """Carga una imagen (JPG, PNG, AVIF, WEBP) y la redimensiona a un cuadrado"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Intentar cargar con PIL (soporta AVIF, WEBP, etc.)
    try:
        pil_img = Image.open(path)
        pil_img = pil_img.convert('RGB')  # Convertir a RGB si es necesario
        img = np.array(pil_img)
    except Exception as e:
        # Fallback a OpenCV (solo JPG, PNG)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Recortar al cuadrado central
    h, w, _ = img.shape
    s = min(h, w)
    cy, cx = h//2, w//2
    img = img[cy - s//2: cy - s//2 + s, cx - s//2: cx - s//2 + s]
    
    # Redimensionar
    img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return img

def build_static_patches(img, N):
    H, W, _ = img.shape
    patch_px = H // N
    patches = []
    
    for r in range(N):
        for c in range(N):
            y0 = r * patch_px
            x0 = c * patch_px
            y1 = y0 + patch_px
            x1 = x0 + patch_px
            pimg = img[y0:y1, x0:x1].copy()
            
            patches.append({
                'id': (r, c),
                'bbox': (x0, y0, x1, y1),
                'img': pimg,
                'match': None
            })
    
    return patches, patch_px

def find_best_square_resized_diff(target_idx, N, ks, num_candidates, patches, img, C_SIDE):
    target_patch = patches[target_idx]
    
    try:
        targ_img_resized = cv2.resize(
            target_patch['img'], (C_SIDE, C_SIDE), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    except:
        return {'top': target_patch['id'], 'k': 1, 'score': 0.0}

    patch_px = img.shape[0] // N
    best = None
    best_score = float('inf')

    for k in ks:
        candidates_r = np.random.randint(0, N - k + 1, num_candidates)
        candidates_c = np.random.randint(0, N - k + 1, num_candidates)
        
        for i in range(num_candidates):
            rt, ct = candidates_r[i], candidates_c[i]
            
            y0, x0 = rt * patch_px, ct * patch_px
            y1, x1 = (rt + k) * patch_px, (ct + k) * patch_px
            
            cand_img_view = img[y0:y1, x0:x1]
            
            try:
                cand_img_resized = cv2.resize(
                    cand_img_view, (C_SIDE, C_SIDE), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
            except:
                continue

            diff_img = np.abs(targ_img_resized - cand_img_resized)
            avg_error_vec = np.mean(diff_img, axis=(0, 1))
            score = np.linalg.norm(avg_error_vec)

            if score < best_score:
                best_score = score
                best_top = (rt, ct)
                best = {'top': best_top, 'k': k, 'score': float(score)}
    
    if best is None:
        best = {'top': target_patch['id'], 'k': 1, 'score': 0.0}
    return best

def candidate_ks(N, max_k=MAX_K):
    ks = []
    upper = min(max_k, N)
    for k in range(upper, 1, -1):
        if (k % 2) == 0:
            ks.append(k)
    if not ks and N >= 2:
        ks = [2]
    return [MAX_K]

def build_static_matches(img, N, max_k, num_candidates, C_SIDE):
    print("\n=== FASE 1: Pre-cálculo de parches ===")
    patches, patch_px = build_static_patches(img, N)
    ks = candidate_ks(N, max_k=max_k)
    total = len(patches)
    
    worker_func = partial(find_best_square_resized_diff,
                          N=N,
                          ks=ks,
                          num_candidates=num_candidates,
                          patches=patches,
                          img=img,
                          C_SIDE=C_SIDE)

    print(f"\n=== FASE 2: Búsqueda de matches ===")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"Grid: {N}x{N} | K values: {ks}")
    print(f"Metric: resized_diff | C={C_SIDE} | Candidates: {num_candidates}")
    
    t0 = time.time()
    matches = [None] * total
    
    with multiprocessing.Pool() as pool:
        results_iterator = pool.imap(worker_func, range(total))
        last_print_time = time.time()
        
        for idx, match_result in enumerate(results_iterator):
            matches[idx] = match_result
            current_time = time.time()
            if current_time - last_print_time > 0.1 or idx == total - 1:
                percentage = (idx + 1) / total * 100
                print(f"  Progreso: {percentage:.1f}% ({idx+1}/{total})", end="\r")
                last_print_time = current_time
        
        print("\n✓ Cálculo de matches completado")
    
    for idx in range(total):
        patches[idx]['match'] = matches[idx]
    
    elapsed = time.time() - t0
    print(f"Tiempo total: {elapsed:.2f}s")
    return patches

def save_project(patches, img, N, project_name, output_dir="output"):
    """Guarda el proyecto en output/<project_name>/ (optimizado: solo imagen original + JSON)"""
    
    print(f"\n=== FASE 3: Guardando proyecto '{project_name}' ===")
    
    # Crear carpeta del proyecto
    project_path = os.path.join(output_dir, project_name)
    os.makedirs(project_path, exist_ok=True)
    
    # Guardar imagen original (la única imagen necesaria)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    source_path = os.path.join(project_path, "source.png")
    cv2.imwrite(source_path, img_bgr)
    print(f"✓ Imagen original guardada: {source_path}")
    
    # ⚠️ NO guardamos patches individuales (se extraen desde source.png usando bbox)
    
    # Preparar datos JSON
    json_data = {
        "metadata": {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "N": N,
            "img_side": IMG_SIDE,
            "patch_count": len(patches),
            "max_k": MAX_K,
            "resize_c_side": RESIZE_C_SIDE,
            "num_candidates": NUM_RANDOM_CANDIDATES
        },
        "patches": []
    }
    
    # Serializar patches (solo coordenadas, no imágenes)
    for patch in patches:
        r, c = patch['id']
        match = patch['match']
        
        patch_data = {
            "id": [int(r), int(c)],
            "bbox": [int(x) for x in patch['bbox']],  # Coordenadas para extraer desde source.png
            "match": {
                "top": [int(x) for x in match['top']] if match else None,
                "k": int(match['k']) if match else 1,
                "score": float(match['score']) if match else 0.0
            } if match else None
        }
        json_data["patches"].append(patch_data)
    
    # Guardar JSON
    json_path = os.path.join(project_path, "project.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadatos guardados: {json_path}")
    print(f"✓ Tamaño optimizado: 1 imagen en lugar de {len(patches)} patches individuales")
    print(f"\n✅ Proyecto '{project_name}' generado exitosamente en: {project_path}")
    
    return project_path

def generate_fractal(image_path, project_name=None, N=None, max_k=None, num_candidates=None, c_side=None):
    """Función principal de generación"""
    
    # Usar valores por defecto si no se especifican
    N = N or INIT_N
    max_k = max_k or MAX_K
    num_candidates = num_candidates or NUM_RANDOM_CANDIDATES
    c_side = c_side or RESIZE_C_SIDE
    
    # Generar nombre del proyecto si no se especifica
    if project_name is None:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"{basename}_{timestamp}"
    
    print("=" * 60)
    print("GENERADOR DE FRACTALES - v7.0")
    print("=" * 60)
    print(f"Imagen: {image_path}")
    print(f"Proyecto: {project_name}")
    print("=" * 60)
    
    # Cargar imagen
    img = load_image_square(image_path, side=IMG_SIDE)
    print(f"✓ Imagen cargada: {img.shape}")
    
    # Procesar patches
    patches = build_static_matches(img, N, max_k, num_candidates, c_side)
    
    # Guardar proyecto
    project_path = save_project(patches, img, N, project_name)
    
    return project_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python generate.py <imagen.jpg> [nombre_proyecto]")
        print("\nEjemplo: python generate.py fractal.jpg mi_fractal")
        sys.exit(1)
    
    image_path = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra el archivo '{image_path}'")
        sys.exit(1)
    
    multiprocessing.set_start_method('spawn', force=True)
    generate_fractal(image_path, project_name)
