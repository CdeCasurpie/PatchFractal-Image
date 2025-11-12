"""
Fractal-by-patches - v7.0 (Tu Métrica Resized-Diff)

- Mantiene zoom (rueda) y paneo (clic).
- N se fija al inicio (no hay UP/DOWN).
- Sin efectos de blur/pixelate.
- OPTIMIZACIÓN: Usa la métrica de 'diferencia redimensionada' (MAE).
- ADVERTENCIA: Esta versión es computacionalmente más pesada que v6.0
  porque re-introduce 'cv2.resize' dentro del bucle del worker,
  eliminando la ventaja del Histograma Integral.
"""

import cv2, math, time, random
import numpy as np
import pygame
from pygame.locals import *
import multiprocessing
from functools import partial

# --- CONFIGURACIÓN ELEGANTE ---

# -- Renderizado --
IMG_PATH = "../input.jpg"
IMG_SIDE = 1024          # Lado de la imagen interna (512x512)
CANVAS_PX = 800
LOD_THRESHOLD_PX = 16    # Píxeles para dejar de recursar

# -- Estructura Fractal --
INIT_N = 60             # Grid NxN (Fijo)
MAX_K = 16              # Máx. k a buscar (k=4, k=2)

# -- Búsqueda de Similitud --
# Esta es tu métrica: redimensionar a C x C y calcular el error
RESIZE_C_SIDE = 128 # El 'c' en 'c x c' para la comparación
# Número de bloques aleatorios a probar por cada k
NUM_RANDOM_CANDIDATES = 7000

# -----------------------------

def load_image_square(path, side):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    s = min(h,w)
    cy, cx = h//2, w//2
    img = img[cy - s//2: cy - s//2 + s, cx - s//2: cx - s//2 + s]
    img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return img

def build_static_patches(img, N):
    """
    Pre-cálculo: Solo divide la imagen en parches.
    (Ya no se calculan histogramas).
    """
    H,W,_ = img.shape
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
                'id': (r,c),
                'bbox': (x0,y0,x1,y1), 
                'img': pimg,
                'match': None
            })
            
    return patches, patch_px

def find_best_square_resized_diff(target_idx, N, ks, num_candidates, patches, img, C_SIDE):
    """
    Worker v7.0: Implementa tu métrica de diferencia redimensionada.
    """
    target_patch = patches[target_idx]
    
    # 1. Redimensionar el parche objetivo UNA VEZ
    try:
        targ_img_resized = cv2.resize(
            target_patch['img'], (C_SIDE, C_SIDE), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
    except:
        # Fallback si el parche está vacío
        return {'top': target_patch['id'], 'k':1, 'score': 0.0}

    patch_px = img.shape[0] // N
    best = None
    best_score = float('inf') # Queremos minimizar esta puntuación (distancia a 0)

    for k in ks:
        # 2. Generar candidatos aleatorios
        candidates_r = np.random.randint(0, N - k + 1, num_candidates)
        candidates_c = np.random.randint(0, N - k + 1, num_candidates)
        
        for i in range(num_candidates):
            rt, ct = candidates_r[i], candidates_c[i]
            
            # 3. Obtener y redimensionar el bloque candidato (¡COSTOSO!)
            y0, x0 = rt * patch_px, ct * patch_px
            y1, x1 = (rt + k) * patch_px, (ct + k) * patch_px
            
            cand_img_view = img[y0:y1, x0:x1]
            
            try:
                cand_img_resized = cv2.resize(
                    cand_img_view, (C_SIDE, C_SIDE), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
            except:
                continue # Omitir si el candidato es inválido

            # 4. Calcular diferencia matricial
            diff_img = np.abs(targ_img_resized - cand_img_resized)
            
            # 5. Calcular el vector de error promedio
            avg_error_vec = np.mean(diff_img, axis=(0, 1))
            
            # 6. Puntuación = Distancia de ese vector a [0, 0, 0]
            score = np.linalg.norm(avg_error_vec)

            # 7. Minimizar la puntuación
            if score < best_score:
                best_score = score
                best_top = (rt, ct)
                best = {'top': best_top, 'k':k, 'score':float(score)}
    
    if best is None:
        best = {'top': target_patch['id'], 'k':1, 'score': 0.0}
    return best


def build_static_matches(img, N, max_k, num_candidates, C_SIDE):
    print("Fase 1: Pre-cálculo de parches...")
    patches, patch_px = build_static_patches(img, N)
    ks = candidate_ks(N, max_k=max_k)
    total = len(patches)
    
    # 2. Configurar el worker de multiprocessing
    worker_func = partial(find_best_square_resized_diff,
                          N=N,
                          ks=ks,
                          num_candidates=num_candidates,
                          patches=patches,
                          img=img,
                          C_SIDE=C_SIDE)

    print(f"Fase 2: Búsqueda de matches (paralelizada en {multiprocessing.cpu_count()} núcleos)...")
    print(f"   (N={N}, k={ks}, metric=resized_diff, C={C_SIDE}, candidates={num_candidates})")
    print(f"   ADVERTENCIA: Esta métrica es lenta.")
    t0 = time.time()
    matches = [None] * total
    
    with multiprocessing.Pool() as pool:
        results_iterator = pool.imap(worker_func, range(total))
        last_print_time = time.time()
        print() 

        for idx, match_result in enumerate(results_iterator):
            matches[idx] = match_result
            current_time = time.time()
            if current_time - last_print_time > 0.1 or idx == total - 1:
                percentage = (idx + 1) / total * 100
                print(f"  -> Calculando... {percentage:.1f}% ({idx+1}/{total})", end="\r")
                last_print_time = current_time
        
        print() 
        print("Cálculo de matches completado.")

    
    for idx in range(total):
        patches[idx]['match'] = matches[idx]
        
    print(f"Matches construidos en {time.time()-t0:.2f}s")
    return patches

def candidate_ks(N, max_k=MAX_K):
    ks = []
    upper = min(max_k, N)
    for k in range(upper, upper, -1):
        if (k % 2)==0:
            ks.append(k)
    if not ks and N >= 2:
        ks = [2]
    return ks

def render_dynamic_zoom(surface, patches, N, view_rect, lod_threshold):
    surface.fill((0,0,0))
    vx, vy, vw, vh = view_rect
    
    def blit_patch_img(screen_bbox, pimg):
        sx, sy, sw, sh = screen_bbox
        
        x0 = round(sx)
        y0 = round(sy)
        x1 = round(sx + sw)
        y1 = round(sy + sh)
        
        w = x1 - x0
        h = y1 - y0
        
        if w <= 0 or h <= 0: return
        
        try:
            interp = cv2.INTER_NEAREST 
            patch_resized = cv2.resize(pimg, (w,h), interpolation=interp)
            surf = pygame.surfarray.make_surface(patch_resized.swapaxes(0,1))
            surface.blit(surf, (x0, y0))
        except Exception as e:
            pass

    def render_node(r, c, norm_bbox):
        nx, ny, nw, nh = norm_bbox
        
        if (nx + nw < vx or nx > vx + vw or 
            ny + nh < vy or ny > vy + vh):
            return

        sx = (nx - vx) / vw * CANVAS_PX
        sy = (ny - vy) / vh * CANVAS_PX
        sw = nw / vw * CANVAS_PX
        sh = nh / vw * CANVAS_PX
        screen_bbox = (sx, sy, sw, sh)

        idx = r * N + c
        if idx >= len(patches): return
        node = patches[idx]
        
        # Usar el parámetro lod_threshold en lugar de la constante global
        if (sw < lod_threshold or sh < lod_threshold or 
            node['match'] is None or node['match']['k'] <= 1):
            
            blit_patch_img(screen_bbox, node['img'])
            return

        match = node['match']
        k = match['k']
        top_r, top_c = match['top']
        
        child_nw = nw / k
        child_nh = nh / k

        for rr in range(k):
            for cc in range(k):
                child_r = top_r + rr
                child_c = top_c + cc
                
                if child_r < 0 or child_r >= N or child_c < 0 or child_c >= N:
                    continue
                
                child_nx = nx + cc * child_nw
                child_ny = ny + rr * child_nh
                child_norm_bbox = (child_nx, child_ny, child_nw, child_nh)
                
                render_node(child_r, child_c, child_norm_bbox)

    norm_patch_w = 1.0 / N
    norm_patch_h = 1.0 / N
    
    for r in range(N):
        for c in range(N):
            nx = c * norm_patch_w
            ny = r * norm_patch_h
            norm_bbox = (nx, ny, norm_patch_w, norm_patch_h)
            render_node(r, c, norm_bbox)

def print_static_matches(patches):
    print("STATIC PATCHES MATCHES:")
    metric_name = "resized_diff_error"
    for p in patches:
        r,c = p['id'];
        m = p['match']
        score_str = f"{m['score']:.3f}" if m and 'score' in m else "N/A"
        k_str = m['k'] if m and 'k' in m else "N/A"
        top_str = m['top'] if m and 'top' in m else "N/A"
        print(f"id {(r,c)} -> top {top_str} k {k_str} score {score_str} ({metric_name})")

def main():
    img = load_image_square(IMG_PATH, side=IMG_SIDE)
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_PX, CANVAS_PX))
    surf_img = pygame.Surface((CANVAS_PX, CANVAS_PX))
    clock = pygame.time.Clock()

    N = INIT_N
    
    # Variable local para el LOD threshold (puede cambiar en tiempo real)
    lod_threshold_px = LOD_THRESHOLD_PX
    
    view_rect = (0.0, 0.0, 1.0, 1.0)
    panning = False
    last_pan_pos = (0, 0)

    patches = build_static_matches(img, N,
                                   max_k=MAX_K,
                                   num_candidates=NUM_RANDOM_CANDIDATES,
                                   C_SIDE=RESIZE_C_SIDE)

    running = True
    dirty = True
    while running:
        
        vx, vy, vw, vh = view_rect
        
        events = pygame.event.get()
        for ev in events:
            if ev.type == QUIT:
                running = False
            
            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_s:
                    print_static_matches(patches)
                    dirty = False # No necesita re-renderizar
                
                # Teclas para controlar el nivel de detalle (profundidad)
                elif ev.key == K_u or ev.key == K_UP:
                    # Aumentar LOD threshold = Menos profundidad (más rápido)
                    lod_threshold_px = min(200, lod_threshold_px + 2)
                    print(f"LOD Threshold: {lod_threshold_px}px (menos detalle/profundidad)")
                    dirty = True
                
                elif ev.key == K_j or ev.key == K_DOWN:
                    # Disminuir LOD threshold = Más profundidad (más lento)
                    lod_threshold_px = max(1, lod_threshold_px - 2)
                    print(f"LOD Threshold: {lod_threshold_px}px (más detalle/profundidad)")
                    dirty = True
                
            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    panning = True
                    last_pan_pos = ev.pos
                elif ev.button == 4: # Zoom In
                    zoom_factor = 0.8
                    mx = ev.pos[0] / CANVAS_PX
                    my = ev.pos[1] / CANVAS_PX
                    img_mx = vx + mx * vw
                    img_my = vy + my * vh
                    vw_new = vw * zoom_factor
                    vh_new = vh * zoom_factor
                    vx = img_mx - mx * vw_new
                    vy = img_my - my * vh_new
                    vw = vw_new
                    vh = vh_new
                    dirty = True
                elif ev.button == 5: # Zoom Out
                    zoom_factor = 1.25
                    mx = ev.pos[0] / CANVAS_PX
                    my = ev.pos[1] / CANVAS_PX
                    img_mx = vx + mx * vw
                    img_my = vy + my * vh
                    vw_new = vw * zoom_factor
                    vh_new = vh * zoom_factor
                    vx = img_mx - mx * vw_new
                    vy = img_my - my * vh_new
                    vw = vw_new
                    vh = vh_new
                    dirty = True

            elif ev.type == MOUSEBUTTONUP:
                if ev.button == 1:
                    panning = False

            elif ev.type == MOUSEMOTION:
                if panning:
                    dx, dy = ev.pos[0] - last_pan_pos[0], ev.pos[1] - last_pan_pos[1]
                    last_pan_pos = ev.pos
                    norm_dx = (dx / CANVAS_PX) * vw
                    norm_dy = (dy / CANVAS_PX) * vh
                    vx -= norm_dx
                    vy -= norm_dy
                    dirty = True
        
        # Límites de zoom y paneo (v6.1)
        vw = max(0, min(1.0, vw)) # Zoom "infinito"
        vh = vw 
        vx = max(0.0, min(1.0 - vw, vx)) # Paneo lógico
        vy = max(0.0, min(1.0 - vh, vy))
        view_rect = (vx, vy, vw, vh)

        if dirty:
            t_render_start = time.time()
            
            render_dynamic_zoom(surf_img, patches, N, view_rect, lod_threshold_px)

            screen.blit(surf_img, (0,0))
            
            pygame.display.set_caption(f"N={N} | Zoom: {1/vw:.2f}x | LOD: {lod_threshold_px}px | Metric: Resized-Diff (C={RESIZE_C_SIDE})")
            pygame.display.flip()
            dirty = False
            
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()