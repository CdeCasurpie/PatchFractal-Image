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
IMG_PATH = "./fractal.jpg"
IMG_SIDE = 1024          # Lado de la imagen interna (512x512)
CANVAS_PX = 800
LOD_THRESHOLD_PX = 20    # Píxeles para dejar de recursar

# -- Estructura Fractal --
INIT_N = 15             # Grid NxN (Fijo)
MAX_K = 16              # Máx. k a buscar (k=4, k=2)

# -- Búsqueda de Similitud --
# Esta es tu métrica: redimensionar a C x C y calcular el error
RESIZE_C_SIDE = 128 # El 'c' en 'c x c' para la comparación
# Número de bloques aleatorios a probar por cada k
NUM_RANDOM_CANDIDATES = 1500

# -- Blending Seamless --
BLEND_ENABLED = True     # Activar/desactivar blending
BLEND_MARGIN = 4         # Píxeles de margen para el blending (mask_size)

# -- Ajuste de Brillo --
BRIGHTNESS_BOOST = 1.15  # Factor de brillo inicial (1.0 = sin cambio, >1.0 = más brillante)

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

def adjust_brightness(img, factor):
    """
    Ajusta el brillo de una imagen multiplicando los valores de píxeles.
    
    Args:
        img: Imagen numpy array (RGB)
        factor: Factor de brillo (1.0 = sin cambio, >1.0 = más brillante, <1.0 = más oscuro)
    
    Returns:
        Imagen con brillo ajustado
    """
    if factor == 1.0:
        return img
    
    # Convertir a float, aplicar factor y recortar valores
    adjusted = img.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

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

def create_blend_mask(h, w, margin):
    """
    Crea una máscara de transición suave usando una función coseno.
    Los bordes se atenúan gradualmente desde 0 (borde) hasta 1 (centro).
    """
    mask = np.ones((h, w), dtype=np.float32)
    
    if margin <= 0 or margin >= min(h, w) // 2:
        return mask
    
    # Crear rampas para cada borde
    for i in range(margin):
        # Factor de transición suave (coseno)
        alpha = (1 - np.cos(i * np.pi / margin)) / 2
        
        # Aplicar a los 4 bordes
        mask[i, :] = np.minimum(mask[i, :], alpha)           # Top
        mask[h-1-i, :] = np.minimum(mask[h-1-i, :], alpha)   # Bottom
        mask[:, i] = np.minimum(mask[:, i], alpha)           # Left
        mask[:, w-1-i] = np.minimum(mask[:, w-1-i], alpha)   # Right
    
    return mask

def blend_patches_seamless(patch_img, base_img, mask_size=8):
    """
    Fusiona suavemente un patch sobre una imagen base usando una máscara de transición.
    
    Args:
        patch_img: Imagen del patch a fusionar (numpy array RGB)
        base_img: Imagen base sobre la que fusionar (numpy array RGB, mismo tamaño)
        mask_size: Tamaño del margen de transición en píxeles
    
    Returns:
        Imagen fusionada con bordes suaves (numpy array RGB)
    """
    if patch_img.shape != base_img.shape:
        # Si las dimensiones no coinciden, redimensionar base_img
        base_img = cv2.resize(base_img, (patch_img.shape[1], patch_img.shape[0]), 
                             interpolation=cv2.INTER_LINEAR)
    
    h, w = patch_img.shape[:2]
    
    # Crear máscara de transición
    mask = create_blend_mask(h, w, mask_size)
    
    # Expandir máscara a 3 canales
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    
    # Blending: patch * mask + base * (1 - mask)
    blended = (patch_img.astype(np.float32) * mask_3ch + 
               base_img.astype(np.float32) * (1 - mask_3ch))
    
    return blended.astype(np.uint8)

def render_dynamic_zoom(surface, patches, N, view_rect, lod_threshold, blend_enabled=True, blend_margin=8, brightness_factor=1.0):
    surface.fill((0,0,0))
    vx, vy, vw, vh = view_rect
    
    # Canvas temporal para acumular patches con blending
    if blend_enabled:
        canvas_array = np.zeros((CANVAS_PX, CANVAS_PX, 3), dtype=np.uint8)
    
    def blit_patch_img(screen_bbox, pimg, apply_blend=False):
        sx, sy, sw, sh = screen_bbox
        
        x0 = round(sx)
        y0 = round(sy)
        x1 = round(sx + sw)
        y1 = round(sy + sh)
        
        w = x1 - x0
        h = y1 - y0
        
        if w <= 0 or h <= 0: return
        
        try:
            interp = cv2.INTER_LINEAR if blend_enabled else cv2.INTER_NEAREST
            patch_resized = cv2.resize(pimg, (w,h), interpolation=interp)
            
            # Aplicar ajuste de brillo
            if brightness_factor != 1.0:
                patch_resized = adjust_brightness(patch_resized, brightness_factor)
            
            if blend_enabled and apply_blend:
                # Aplicar blending con el contenido existente
                nonlocal canvas_array
                
                # Extraer región de la base
                y0_clip = max(0, y0)
                y1_clip = min(CANVAS_PX, y1)
                x0_clip = max(0, x0)
                x1_clip = min(CANVAS_PX, x1)
                
                if y1_clip > y0_clip and x1_clip > x0_clip:
                    # Recortar patch si está parcialmente fuera
                    patch_y0 = y0_clip - y0
                    patch_y1 = patch_y0 + (y1_clip - y0_clip)
                    patch_x0 = x0_clip - x0
                    patch_x1 = patch_x0 + (x1_clip - x0_clip)
                    
                    patch_region = patch_resized[patch_y0:patch_y1, patch_x0:patch_x1]
                    base_region = canvas_array[y0_clip:y1_clip, x0_clip:x1_clip]
                    
                    # Calcular margen adaptativo (proporcional al tamaño del patch)
                    adaptive_margin = min(blend_margin, min(patch_region.shape[0], patch_region.shape[1]) // 4)
                    
                    # Aplicar blending
                    blended_region = blend_patches_seamless(patch_region, base_region, adaptive_margin)
                    canvas_array[y0_clip:y1_clip, x0_clip:x1_clip] = blended_region
            else:
                # Modo sin blending (original)
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
            
            blit_patch_img(screen_bbox, node['img'], apply_blend=blend_enabled)
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
    
    # Si usamos blending, transferir el canvas final a la surface de pygame
    if blend_enabled:
        final_surf = pygame.surfarray.make_surface(canvas_array.swapaxes(0,1))
        surface.blit(final_surf, (0, 0))

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
    
    # Control de blending
    blend_enabled = BLEND_ENABLED
    blend_margin = BLEND_MARGIN
    
    # Control de brillo
    brightness_factor = BRIGHTNESS_BOOST
    
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
                
                # Toggle blending con tecla 'B'
                elif ev.key == K_b:
                    blend_enabled = not blend_enabled
                    print(f"Blending: {'ENABLED' if blend_enabled else 'DISABLED'}")
                    dirty = True
                
                # Ajustar margen de blending con +/-
                elif ev.key == K_PLUS or ev.key == K_EQUALS:
                    blend_margin = min(32, blend_margin + 2)
                    print(f"Blend Margin: {blend_margin}px")
                    dirty = True
                
                elif ev.key == K_MINUS:
                    blend_margin = max(2, blend_margin - 2)
                    print(f"Blend Margin: {blend_margin}px")
                    dirty = True
                
                # Control de brillo con teclas L (aumentar) y K (disminuir)
                elif ev.key == K_l:
                    brightness_factor = min(3.0, brightness_factor + 0.05)
                    print(f"Brillo: {brightness_factor:.2f}x")
                    dirty = True
                
                elif ev.key == K_k:
                    brightness_factor = max(0.3, brightness_factor - 0.05)
                    print(f"Brillo: {brightness_factor:.2f}x")
                    dirty = True
                
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
                    vh_new = vw * zoom_factor
                    vx = img_mx - mx * vw_new
                    vy = img_my - my * vh_new
                    vw = vw_new
                    vh = vw_new
                    dirty = True
                elif ev.button == 5: # Zoom Out
                    zoom_factor = 1.25
                    mx = ev.pos[0] / CANVAS_PX
                    my = ev.pos[1] / CANVAS_PX
                    img_mx = vx + mx * vw
                    img_my = vy + my * vh
                    vw_new = vw * zoom_factor
                    vh_new = vw * zoom_factor
                    vx = img_mx - mx * vw_new
                    vy = img_my - my * vh_new
                    vw = vw_new
                    vh = vw_new
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
            
            render_dynamic_zoom(surf_img, patches, N, view_rect, lod_threshold_px, 
                              blend_enabled=blend_enabled, blend_margin=blend_margin, brightness_factor=brightness_factor)

            screen.blit(surf_img, (0,0))
            
            blend_status = "ON" if blend_enabled else "OFF"
            pygame.display.set_caption(f"N={N} | Zoom: {1/vw:.2f}x | LOD: {lod_threshold_px}px | Blend: {blend_status}({blend_margin}px) | Brillo: {brightness_factor:.2f}x")
            pygame.display.flip()
            dirty = False
            
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()