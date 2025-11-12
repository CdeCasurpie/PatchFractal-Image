"""
Fractal-by-patches - v5.4 (Muestreo/Pixelado)
- Requisitos: pip install numpy opencv-python pygame scikit-image
- Poner una imagen cuadrada llamada 'input.jpg' en la misma carpeta
- Poner el archivo 'fractal_workers.py' en la misma carpeta
- Teclas:
    UP / DOWN -> aumentar / disminuir N (recalcula matches)
    S -> imprime en consola la lista estática de patches y matches
    P -> Activa/desactiva el efecto "Muestreo" (Pixelado)
    ESC -> salir
- Ratón:
    RUEDA (Scroll) -> Zoom In / Zoom Out (dinámico)
    CLICK IZQ + ARRASTRAR -> Pan (mover la vista)
Notas:
 - v5.4 CAMBIOS:
 - 1. Se revirtió el renderizado a la lógica de v5.2. El caso base
      vuelve a ser el 'pimg' de baja resolución ('node['img']').
      ¡El "efecto de bloques" ha vuelto!
 - 2. Se eliminó la lógica de "Hi-Res" (v5.3) que no te gustó.
 - 3. ¡Nuevo Efecto de Muestreo! Presiona 'P' para activar un
      "pixelado" de la imagen final, que es exactamente lo que
      describiste (promedio de color en celdas m x m).
"""
import cv2, math, time, random
import numpy as np
import pygame
from pygame.locals import *
from skimage.color import rgb2lab
import multiprocessing
from functools import partial

# --- ¡NUEVA IMPORTACIÓN! ---
# Asume que 'fractal_workers.py' está en la misma carpeta
from fractal_workers import find_best_square_histogram

# ------------- CONFIG -------------
INPUT_PATH = "input.jpg" # Ajustado para buscar en la misma carpeta
IMG_SIDE = 512          # trabajo interno (square)
CANVAS_PX = 800
INIT_N = 60
MAX_K = 4               # Límite de k (k=2, k=4).

# --- NUEVA CONFIGURACIÓN DE SIMILITUD ---
HIST_BINS_PER_CHANNEL = 16
SIM_METRIC = "histogram_cosine"

LOD_THRESHOLD_PX = 8    # Umbral de píxeles para dejar de recursar
NUM_RANDOM_CANDIDATES = 250

# --- NUEVA CONFIGURACIÓN DE MUESTREO (PIXELADO) ---
# Este es el 'm x m' que mencionaste.
PIXELATE_GRID_SIZE = 50 # (ej. 100x100 celdas)
# -----------------------------------

def load_image_square(path, side=IMG_SIDE):
    img = cv2.imread(path)
    if img is None:
        try:
            img = cv2.imread("../" + path)
        except:
            pass
        
    if img is None:
        raise FileNotFoundError(f"File not found: {path} (or ../{path})")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    s = min(h,w)
    cy, cx = h//2, w//2
    img = img[cy - s//2: cy - s//2 + s, cx - s//2: cx - s//2 + s]
    img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return img

def calculate_histogram(img, bins_per_channel):
    """
    Calcula el histograma de color (vector de características) para una imagen.
    Devuelve un vector de recuentos crudos de tamaño (bins_per_channel^3).
    """
    if img is None or img.size == 0:
        return np.zeros(bins_per_channel ** 3, dtype=np.int32)
    
    # 1. Discretizar
    bin_size = 256.0 / bins_per_channel
    img_discrete = (img / bin_size).astype(np.int32)
    
    # 2. Asegurar que esté en rango [0, bins-1]
    img_discrete = np.clip(img_discrete, 0, bins_per_channel - 1)
    
    # 3. Convertir (R, G, B) bin-indices a un índice de bin lineal (único)
    r, g, b = img_discrete[:,:,0], img_discrete[:,:,1], img_discrete[:,:,2]
    linear_bins = r * (bins_per_channel**2) + g * bins_per_channel + b
    
    # 4. Contar ocurrencias de cada bin lineal
    total_bins = bins_per_channel ** 3
    hist = np.bincount(linear_bins.ravel(), minlength=total_bins)
    return hist.astype(np.int32)

def build_static_patches(img, N, bins_per_channel):
    """
    Pre-cálculo: Divide la imagen y calcula el histograma
    (vector de características) para cada parche.
    """
    print(f"Pre-calculando histogramas de {N*N} parches...")
    H,W,_ = img.shape
    assert H==W
    patch_px = H // N
    patches = []
    
    total_bins = bins_per_channel ** 3
    
    for r in range(N):
        for c in range(N):
            y0 = r * patch_px
            x0 = c * patch_px
            y1 = y0 + patch_px
            x1 = x0 + patch_px
            pimg = img[y0:y1, x0:x1].copy()
            
            # --- PRE-CÁLCULO ---
            hist_vec_raw = calculate_histogram(pimg, bins_per_channel)
            
            if hist_vec_raw.shape[0] != total_bins:
                 hist_vec_raw = np.zeros(total_bins, dtype=np.int32)
            
            patches.append({
                'id': (r,c),
                'bbox': (x0,y0,x1,y1), 
                'img': pimg, # <-- Se usa para el renderizado de baja resolución
                'hist_vec_raw': hist_vec_raw,
                'match': None
            })
    print("Histogramas pre-calculados.")
    return patches, patch_px

def candidate_ks(patch_px, N, max_k=MAX_K):
    ks = []
    upper = min(max_k, N)
    for k in range(upper, 1, -1):
        if (k % 2)==0:
            ks.append(k)
    if not ks and N >= 2:
        ks = [2]
    return ks

# 'find_best_square_histogram' se importa desde 'fractal_workers.py'

def build_static_matches(img, N, bins_per_channel, max_k=MAX_K, metric=SIM_METRIC, num_candidates=NUM_RANDOM_CANDIDATES):
    # 1. Pre-cálculo de histogramas
    patches, patch_px = build_static_patches(img, N, bins_per_channel)
    ks = candidate_ks(patch_px, N, max_k=max_k)
    
    if len(ks)==0:
        print(f"Advertencia: No se encontró k par válido para N={N} y max_k={max_k}. Usando k=1.")
        for idx in range(len(patches)):
             patches[idx]['match'] = {'top': patches[idx]['id'], 'k':1, 'score': 0.0}
        return patches, patch_px

    total = len(patches)
    t0 = time.time()

    # 2. Configurar el worker de multiprocessing
    worker_func = partial(find_best_square_histogram,
                          patches=patches,
                          N=N,
                          ks=ks,
                          metric=metric,
                          num_candidates=num_candidates)

    with multiprocessing.Pool() as pool:
        print(f"Iniciando cálculo de matches (N={N}, k={ks}, metric={metric}, bins={bins_per_channel**3}) en {multiprocessing.cpu_count()} núcleos...")
        
        matches = [None] * total
        results_iterator = pool.imap(worker_func, range(total))
        last_print_time = time.time()
        print() 

        # 3. Recolectar resultados con barra de progreso
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
    return patches, patch_px


# ------------ RENDER v5.4 (Render Lo-Res) -------------

def render_dynamic_zoom(surface, patches, N, view_rect):
    """
    Renderiza usando los parches de baja resolución ('pimg')
    para crear el "efecto de bloques".
    """
    surface.fill((0,0,0))
    vx, vy, vw, vh = view_rect
    
    def blit_patch_img(screen_bbox, pimg):
        """
        Dibuja el 'pimg' de baja resolución.
        """
        sx, sy, sw, sh = screen_bbox # Son floats (x, y, w, h)
        
        # --- v5.2 CORRECCIÓN DE BORDES NEGROS ---
        x0 = round(sx)
        y0 = round(sy)
        x1 = round(sx + sw)
        y1 = round(sy + sh)
        
        w = x1 - x0
        h = y1 - y0
        # -----------------------------------
        
        if w <= 0 or h <= 0: return
        
        try:
            # INTER_NEAREST es crucial para el look "pixelado"
            interp = cv2.INTER_NEAREST 
            patch_resized = cv2.resize(pimg, (w,h), interpolation=interp)
            surf = pygame.surfarray.make_surface(patch_resized.swapaxes(0,1))
            surface.blit(surf, (x0, y0)) # Usar las coordenadas de píxel (x0, y0)
        except Exception as e:
            pass # Ignorar errores de resize

    def render_node(r, c, norm_bbox):
        nx, ny, nw, nh = norm_bbox
        
        # 1. Culling
        if (nx + nw < vx or nx > vx + vw or 
            ny + nh < vy or ny > vy + vh):
            return

        # 2. Calcular BBox en Pantalla
        sx = (nx - vx) / vw * CANVAS_PX
        sy = (ny - vy) / vh * CANVAS_PX
        sw = nw / vw * CANVAS_PX
        sh = nh / vh * CANVAS_PX
        screen_bbox = (sx, sy, sw, sh)

        # 3. Decisión de LOD (Nivel de Detalle)
        idx = r * N + c
        if idx >= len(patches): return
        node = patches[idx]
        
        # --- v5.4 CASO BASE (Como v5.2) ---
        # Dibujar el parche de baja resolución 'node['img']'
        if (sw < LOD_THRESHOLD_PX or sh < LOD_THRESHOLD_PX or 
            node['match'] is None or node['match']['k'] <= 1):
            
            blit_patch_img(screen_bbox, node['img'])
            return

        # 4. Recursión (Sin cambios)
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

    # --- Bucle Principal de Renderizado ---
    norm_patch_w = 1.0 / N
    norm_patch_h = 1.0 / N
    
    for r in range(N):
        for c in range(N):
            nx = c * norm_patch_w
            ny = r * norm_patch_h
            norm_bbox = (nx, ny, norm_patch_w, norm_patch_h)
            render_node(r, c, norm_bbox)

def apply_pixelate_effect(surface, grid_size_m):
    """
    Toma la 'surface' renderizada y aplica un efecto de "muestreo"
    de m x m celdas, promediando el color de cada una.
    """
    canvas_size = surface.get_width()
    if grid_size_m <= 0:
        return # No hacer nada
        
    cell_size = canvas_size / grid_size_m # Ahora es flotante

    for i in range(grid_size_m):
        for j in range(grid_size_m):
            # Calcular el bbox de la celda en píxeles
            x0 = round(i * cell_size)
            y0 = round(j * cell_size)
            x1 = round((i + 1) * cell_size)
            y1 = round((j + 1) * cell_size)
            
            w = x1 - x0
            h = y1 - y0

            if w <= 0 or h <= 0:
                continue
                
            # Crear el Rect para esta celda
            sub_rect = pygame.Rect(x0, y0, w, h)
            
            # ¡La magia! Obtener el color promedio de esa área
            try:
                avg_color = pygame.transform.average_color(surface, sub_rect)
                # Dibujar un rectángulo sólido con ese color promedio
                pygame.draw.rect(surface, avg_color, sub_rect)
            except Exception as e:
                # Puede fallar si la sub_rect está fuera de los límites
                pass

def print_static_matches(patches):
    print("STATIC PATCHES MATCHES:")
    metric_name = SIM_METRIC.split('_')[-1] # 'cosine' o 'euclidean'
    for p in patches:
        r,c = p['id'];
        m = p['match']
        score_str = f"{m['score']:.3f}" if m and 'score' in m else "N/A"
        k_str = m['k'] if m and 'k' in m else "N/A"
        top_str = m['top'] if m and 'top' in m else "N/A"
        print(f"id {(r,c)} -> top {top_str} k {k_str} score {score_str} ({metric_name})")

# ---------------- MAIN ----------------
def main():
    img = load_image_square(INPUT_PATH, side=IMG_SIDE)
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_PX, CANVAS_PX))
    surf_img = pygame.Surface((CANVAS_PX, CANVAS_PX))
    clock = pygame.time.Clock()

    N = INIT_N
    
    view_rect = (0.0, 0.0, 1.0, 1.0)
    panning = False
    last_pan_pos = (0, 0)
    pixelate_enabled = False # Nuevo estado para el Muestreo

    print("Building static patch matches (this may take a few seconds)...")
    patches, patch_px = build_static_matches(img, N,
                                             bins_per_channel=HIST_BINS_PER_CHANNEL,
                                             max_k=MAX_K,
                                             metric=SIM_METRIC,
                                             num_candidates=NUM_RANDOM_CANDIDATES)

    running = True
    dirty = True
    while running:
        
        vx, vy, vw, vh = view_rect
        
        events = pygame.event.get()
        for ev in events:
            if ev.type == QUIT:
                running = False
            
            elif ev.type == KEYDOWN:
                dirty = True
                if ev.key == K_ESCAPE:
                    running = False
                elif ev.key == K_UP:
                    N = min(150, N+10)
                    print(f"Recalculando matches para N={N}...")
                    patches, patch_px = build_static_matches(img, N, bins_per_channel=HIST_BINS_PER_CHANNEL, max_k=MAX_K, metric=SIM_METRIC, num_candidates=NUM_RANDOM_CANDIDATES)
                elif ev.key == K_DOWN:
                    N = max(2, N-10)
                    print(f"Recalculando matches para N={N}...")
                    patches, patch_px = build_static_matches(img, N, bins_per_channel=HIST_BINS_PER_CHANNEL, max_k=MAX_K, metric=SIM_METRIC, num_candidates=NUM_RANDOM_CANDIDATES)
                elif ev.key == K_s:
                    print_static_matches(patches)
                elif ev.key == K_p: # Nueva tecla 'P' para PIXELAR
                    pixelate_enabled = not pixelate_enabled
                    print(f"Efecto Muestreo/Pixelado: {'Activado' if pixelate_enabled else 'Desactivado'}")
                else:
                    dirty = False

            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    panning = True
                    last_pan_pos = ev.pos
                elif ev.button == 4: # Rueda Arriba (Zoom In)
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
                elif ev.button == 5: # Rueda Abajo (Zoom Out)
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
        
        vw = max(0.0001, min(10.0, vw))
        vh = vw 
        vx = max(-1.0, min(2.0 - vw, max(-1.0, vx)))
        vy = max(-1.0, min(2.0 - vh, max(-1.0, vy)))
        view_rect = (vx, vy, vw, vh)

        if dirty:
            t_render_start = time.time()
            
            # 1. Renderizar el fractal (de baja resolución)
            render_dynamic_zoom(surf_img, patches, N, view_rect)

            # 2. Aplicar Muestreo/Pixelado si está activado
            if pixelate_enabled:
                apply_pixelate_effect(surf_img, PIXELATE_GRID_SIZE)

            # 3. Dibujar el resultado final en la pantalla
            screen.blit(surf_img, (0,0))
            
            pixel_status = f"PIXELATE({PIXELATE_GRID_SIZE})" if pixelate_enabled else ""
            pygame.display.set_caption(f"N={N} | Zoom: {1/vw:.2f}x | Metric: {SIM_METRIC} | {pixel_status}")
            pygame.display.flip()
            # print(f"Renderizado completo en {time.time()-t_render_start:.3f}s")
            dirty = False
            
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    # Corregir el 'deadlock' de multiprocessing + pygame (de v4.1)
    multiprocessing.set_start_method('spawn')
    main()