"""
render.py - Renderizado interactivo de proyectos fractales
Carga un proyecto desde /output/<nombre>/ y lo renderiza con zoom/pan
"""

import cv2
import numpy as np
import json
import os
import pygame
from pygame.locals import *

# --- CONFIGURACIÓN ---
CANVAS_PX = 800
LOD_THRESHOLD_PX = 20
BLEND_ENABLED = True
BLEND_MARGIN = 4
BRIGHTNESS_BOOST = 1.15

def list_projects(output_dir="output"):
    """Lista todos los proyectos disponibles en /output/"""
    if not os.path.exists(output_dir):
        return []
    
    projects = []
    for item in os.listdir(output_dir):
        project_path = os.path.join(output_dir, item)
        if os.path.isdir(project_path):
            json_path = os.path.join(project_path, "project.json")
            if os.path.exists(json_path):
                projects.append(item)
    
    return sorted(projects)

def select_project(output_dir="output"):
    """Permite al usuario seleccionar un proyecto"""
    projects = list_projects(output_dir)
    
    if not projects:
        print("❌ No hay proyectos en /output/")
        print("   Primero genera un proyecto con: python generate.py <imagen.jpg>")
        return None
    
    print("\n" + "=" * 60)
    print("PROYECTOS DISPONIBLES:")
    print("=" * 60)
    
    for i, project in enumerate(projects, 1):
        project_path = os.path.join(output_dir, project)
        json_path = os.path.join(project_path, "project.json")
        
        # Leer metadata
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                created = metadata.get('created_at', 'N/A')
                N = metadata.get('N', 'N/A')
                patches = metadata.get('patch_count', 'N/A')
                
            print(f"{i}. {project}")
            print(f"   Creado: {created}")
            print(f"   Grid: {N}x{N} | Patches: {patches}")
        except:
            print(f"{i}. {project}")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelecciona un proyecto (número o nombre): ").strip()
            
            # Si es un número
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(projects):
                    return projects[idx]
            
            # Si es un nombre
            if choice in projects:
                return choice
            
            print("❌ Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\nCancelado por el usuario.")
            return None

def load_project(project_name, output_dir="output"):
    """Carga un proyecto desde disco (optimizado: extrae patches desde source.png)"""
    project_path = os.path.join(output_dir, project_name)
    json_path = os.path.join(project_path, "project.json")
    source_path = os.path.join(project_path, "source.png")
    
    print(f"\n📂 Cargando proyecto: {project_name}")
    
    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    N = metadata['N']
    
    print(f"   Grid: {N}x{N}")
    print(f"   Patches: {metadata['patch_count']}")
    
    # Cargar imagen original (una sola vez)
    print(f"   Cargando imagen original...")
    source_img = cv2.imread(source_path)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    print(f"   ✓ Imagen cargada: {source_img.shape}")
    
    # Extraer patches desde la imagen original usando bbox
    patches = []
    for patch_data in data['patches']:
        r, c = patch_data['id']
        x0, y0, x1, y1 = patch_data['bbox']
        
        # Extraer región de la imagen original (sin cargar archivo)
        patch_img = source_img[y0:y1, x0:x1].copy()
        
        match_data = patch_data.get('match')
        match = None
        if match_data:
            match = {
                'top': tuple(match_data['top']) if match_data['top'] else (r, c),
                'k': match_data['k'],
                'score': match_data['score']
            }
        
        patches.append({
            'id': (r, c),
            'bbox': (x0, y0, x1, y1),
            'img': patch_img,
            'match': match
        })
    
    print(f"✓ Proyecto cargado: {len(patches)} patches extraídos desde source.png")
    print(f"✓ Tiempo de carga: ~100x más rápido (1 archivo vs {len(patches)} archivos)")
    
    return patches, N, metadata

def adjust_brightness(img, factor):
    if factor == 1.0:
        return img
    adjusted = img.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

def create_blend_mask(h, w, margin):
    mask = np.ones((h, w), dtype=np.float32)
    
    if margin <= 0 or margin >= min(h, w) // 2:
        return mask
    
    for i in range(margin):
        alpha = (1 - np.cos(i * np.pi / margin)) / 2
        mask[i, :] = np.minimum(mask[i, :], alpha)
        mask[h-1-i, :] = np.minimum(mask[h-1-i, :], alpha)
        mask[:, i] = np.minimum(mask[:, i], alpha)
        mask[:, w-1-i] = np.minimum(mask[:, w-1-i], alpha)  # ← CORREGIDO
    
    return mask

def blend_patches_seamless(patch_img, base_img, mask_size=8):
    if patch_img.shape != base_img.shape:
        base_img = cv2.resize(base_img, (patch_img.shape[1], patch_img.shape[0]), 
                             interpolation=cv2.INTER_LINEAR)
    
    h, w = patch_img.shape[:2]
    mask = create_blend_mask(h, w, mask_size)
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    
    blended = (patch_img.astype(np.float32) * mask_3ch + 
               base_img.astype(np.float32) * (1 - mask_3ch))
    
    return blended.astype(np.uint8)

def render_dynamic_zoom(surface, patches, N, view_rect, lod_threshold, 
                       blend_enabled=True, blend_margin=8, brightness_factor=1.0):
    surface.fill((0, 0, 0))
    vx, vy, vw, vh = view_rect
    
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
        
        if w <= 0 or h <= 0:
            return
        
        try:
            interp = cv2.INTER_LINEAR if blend_enabled else cv2.INTER_NEAREST
            patch_resized = cv2.resize(pimg, (w, h), interpolation=interp)
            
            if brightness_factor != 1.0:
                patch_resized = adjust_brightness(patch_resized, brightness_factor)
            
            if blend_enabled and apply_blend:
                nonlocal canvas_array
                
                y0_clip = max(0, y0)
                y1_clip = min(CANVAS_PX, y1)
                x0_clip = max(0, x0)
                x1_clip = min(CANVAS_PX, x1)
                
                if y1_clip > y0_clip and x1_clip > x0_clip:
                    patch_y0 = y0_clip - y0
                    patch_y1 = patch_y0 + (y1_clip - y0_clip)
                    patch_x0 = x0_clip - x0
                    patch_x1 = patch_x0 + (x1_clip - x0_clip)
                    
                    patch_region = patch_resized[patch_y0:patch_y1, patch_x0:patch_x1]
                    base_region = canvas_array[y0_clip:y1_clip, x0_clip:x1_clip]
                    
                    adaptive_margin = min(blend_margin, min(patch_region.shape[0], patch_region.shape[1]) // 4)
                    blended_region = blend_patches_seamless(patch_region, base_region, adaptive_margin)
                    canvas_array[y0_clip:y1_clip, x0_clip:x1_clip] = blended_region
            else:
                surf = pygame.surfarray.make_surface(patch_resized.swapaxes(0, 1))
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
        if idx >= len(patches):
            return
        node = patches[idx]
        
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
    
    if blend_enabled:
        final_surf = pygame.surfarray.make_surface(canvas_array.swapaxes(0, 1))
        surface.blit(final_surf, (0, 0))

def render_project(project_name):
    """Renderiza un proyecto con Pygame"""
    
    patches, N, metadata = load_project(project_name)
    
    pygame.init()
    screen = pygame.display.set_mode((CANVAS_PX, CANVAS_PX))
    pygame.display.set_caption(f"Fractal Viewer - {project_name}")
    surf_img = pygame.Surface((CANVAS_PX, CANVAS_PX))
    clock = pygame.time.Clock()
    
    lod_threshold_px = LOD_THRESHOLD_PX
    blend_enabled = BLEND_ENABLED
    blend_margin = BLEND_MARGIN
    brightness_factor = BRIGHTNESS_BOOST
    
    view_rect = (0.0, 0.0, 1.0, 1.0)
    panning = False
    last_pan_pos = (0, 0)
    
    print("\n" + "=" * 60)
    print("CONTROLES:")
    print("=" * 60)
    print("  Zoom: Rueda del ratón")
    print("  Pan: Click izquierdo + arrastrar")
    print("  Brillo: L (más) / K (menos)")
    print("  Blending: B (on/off)")
    print("  Margen blend: +/-")
    print("  Detalle (LOD): U/↑ (menos) / J/↓ (más)")
    print("  Salir: ESC")
    print("=" * 60)
    
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
                
                elif ev.key == K_b:
                    blend_enabled = not blend_enabled
                    print(f"Blending: {'ON' if blend_enabled else 'OFF'}")
                    dirty = True
                
                elif ev.key == K_PLUS or ev.key == K_EQUALS:
                    blend_margin = min(32, blend_margin + 2)
                    print(f"Blend Margin: {blend_margin}px")
                    dirty = True
                
                elif ev.key == K_MINUS:
                    blend_margin = max(2, blend_margin - 2)
                    print(f"Blend Margin: {blend_margin}px")
                    dirty = True
                
                elif ev.key == K_l:
                    brightness_factor = min(3.0, brightness_factor + 0.05)
                    print(f"Brillo: {brightness_factor:.2f}x")
                    dirty = True
                
                elif ev.key == K_k:
                    brightness_factor = max(0.3, brightness_factor - 0.05)
                    print(f"Brillo: {brightness_factor:.2f}x")
                    dirty = True
                
                elif ev.key == K_u or ev.key == K_UP:
                    lod_threshold_px = min(200, lod_threshold_px + 2)
                    print(f"LOD: {lod_threshold_px}px (menos detalle)")
                    dirty = True
                
                elif ev.key == K_j or ev.key == K_DOWN:
                    lod_threshold_px = max(1, lod_threshold_px - 2)
                    print(f"LOD: {lod_threshold_px}px (más detalle)")
                    dirty = True
            
            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    panning = True
                    last_pan_pos = ev.pos
                elif ev.button == 4:  # Zoom In
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
                elif ev.button == 5:  # Zoom Out
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
        
        vw = max(0, min(1.0, vw))
        vh = vw
        vx = max(0.0, min(1.0 - vw, vx))
        vy = max(0.0, min(1.0 - vh, vy))
        view_rect = (vx, vy, vw, vh)
        
        if dirty:
            render_dynamic_zoom(surf_img, patches, N, view_rect, lod_threshold_px,
                              blend_enabled=blend_enabled, blend_margin=blend_margin,
                              brightness_factor=brightness_factor)
            
            screen.blit(surf_img, (0, 0))
            
            blend_status = "ON" if blend_enabled else "OFF"
            caption = (f"{project_name} | Zoom: {1/vw:.2f}x | LOD: {lod_threshold_px}px | "
                      f"Blend: {blend_status}({blend_margin}px) | Brillo: {brightness_factor:.2f}x")
            pygame.display.set_caption(caption)
            pygame.display.flip()
            dirty = False
        
        clock.tick(60)
    
    pygame.quit()
    print("\n✓ Viewer cerrado")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = select_project()
        if project_name is None:
            sys.exit(0)
    
    render_project(project_name)
