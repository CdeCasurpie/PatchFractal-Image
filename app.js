 // ==================== BACKGROUND STARS (Sin Cambios) ====================
 const starsCanvas = document.getElementById('stars-canvas');
 const starsCtx = starsCanvas.getContext('2d');
 
 starsCanvas.width = window.innerWidth;
 starsCanvas.height = window.innerHeight;
 
 const stars = [];
 const numStars = 200;
 
 for (let i = 0; i < numStars; i++) {
     stars.push({
         x: Math.random() * starsCanvas.width,
         y: Math.random() * starsCanvas.height,
         radius: Math.random() * 2,
         speed: Math.random() * 0.5 + 0.1,
         opacity: Math.random() * 0.5 + 0.5
     });
 }
 
 function animateStars() {
     starsCtx.clearRect(0, 0, starsCanvas.width, starsCanvas.height);
     
     stars.forEach(star => {
         star.y += star.speed;
         if (star.y > starsCanvas.height) {
             star.y = 0;
             star.x = Math.random() * starsCanvas.width;
         }
         
         starsCtx.fillStyle = `rgba(255, 255, 255, ${star.opacity})`;
         starsCtx.beginPath();
         starsCtx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
         starsCtx.fill();
     });
     
     requestAnimationFrame(animateStars);
 }
 
 animateStars();
 
 // ==================== FRACTAL RENDERER (REFACTORIZADO) ====================
 const canvas = document.getElementById('fractal-canvas');
 const ctx = canvas.getContext('2d');
 
 // Configuración
 let CANVAS_SIZE = 600;
 let project = null;
 let patches = [];
 let patchImages = {};
 let N = 0;
 
 // Estado de la vista
 let viewX = 0, viewY = 0, viewW = 1, viewH = 1;
 let lodThreshold = 40;
 let blendEnabled = true;
 let brightnessFactor = 1.2;
 let borderThickness = 2;  // ← NUEVO: Grosor del borde entre patches
 
 // --- NUEVO ESTADO DE RENDERIZADO OPTIMIZADO ---
 let drawList = [];          // La lista de dibujado cacheada
 let isViewDirty = true;     // Flag para recalcular la vista
 let isCalculating = false;  // Flag para saber si estamos calculando
 let calculationProgress = 0; // Progreso del cálculo (0-100)
 
 // --- SISTEMA DE LOD INTELIGENTE ---
 let lastLODLevel = -1;      // Último nivel de LOD calculado
 let lastViewX = 0, lastViewY = 0; // Última posición de la vista
 const LOD_CHANGE_THRESHOLD = 5;  // Cambio de zoom necesario para recalcular
 const PAN_THRESHOLD = 0.3;  // Movimiento necesario para recalcular (10% del ancho)
 
 // --- BATCH PROCESSING ---
 let batchQueue = [];        // Cola de parches a procesar
 let batchIndex = 0;         // Índice actual en el batch
 const BATCH_SIZE = 100;     // Parches a procesar por chunk
 
 // Estado de interacción
 let isPanning = false;
 let lastX = 0, lastY = 0;
 let touchStartDist = 0;
 
 // Resize canvas
 function resizeCanvas() {
     const screen = document.querySelector('.screen');
     const size = Math.min(screen.clientWidth, screen.clientHeight);
     CANVAS_SIZE = size;
     canvas.width = size;
     canvas.height = size;
     isViewDirty = true; // Invalidar la vista
 }
 
 window.addEventListener('resize', resizeCanvas);
 resizeCanvas(); // Llamada inicial
 
 // ==================== PROJECT LOADING (OPTIMIZADO) ====================
 async function loadProjects() {
     try {
         // ✨ NUEVO: Cargar desde projects.json (compatible con GitHub Pages)
         const response = await fetch('output/projects.json');
         const data = await response.json();
         
         const select = document.getElementById('project-select');
         select.innerHTML = '<option value="">-- Seleccionar --</option>';
         
         data.projects.forEach(projectName => {
             const option = document.createElement('option');
             option.value = projectName;
             option.textContent = projectName;
             select.appendChild(option);
         });

         // cargar el primer proyecto automáticamente
         if (data.projects.length > 0) {
             await loadProject(data.projects[0]);
         }
     } catch (error) {
         console.error('Error loading projects:', error);
         // Fallback: intentar el método antiguo (solo funciona con servidor Python)
         try {
             await loadProjectsFromDirectory();
         } catch (e) {
             console.error('Fallback también falló:', e);
         }
     }
 }
 
 // Método antiguo (solo funciona con servidor Python)
 async function loadProjectsFromDirectory() {
     const response = await fetch('output/');
     const html = await response.text();
     
     const parser = new DOMParser();
     const doc = parser.parseFromString(html, 'text/html');
     const links = doc.querySelectorAll('a');
     
     const select = document.getElementById('project-select');
     select.innerHTML = '<option value="">-- Seleccionar --</option>';
     
     links.forEach(link => {
         const href = link.getAttribute('href');
         if (href && href !== '../' && !href.startsWith('/')) {
             const projectName = href.replace(/\/$/, '');
             const option = document.createElement('option');
             option.value = projectName;
             option.textContent = projectName;
             select.appendChild(option);
         }
     });
 }
 
 async function loadSelectedProject() {
     const select = document.getElementById('project-select');
     const projectName = select.value;
     
     if (!projectName) {
         alert('Selecciona un proyecto');
         return;
     }
     
     await loadProject(projectName);
 }
 
 async function loadProject(projectName) {
     try {
         console.log(`Loading project: ${projectName}`);
         
         // Show loading
         ctx.fillStyle = '#ffffff';
         ctx.fillRect(0, 0, canvas.width, canvas.height);
         ctx.fillStyle = '#ffffff';
         ctx.font = '20px monospace';
         ctx.textAlign = 'center';
         ctx.fillText('LOADING...', canvas.width / 2, canvas.height / 2);
         
         // ✨ OPTIMIZACIÓN: Cargar solo 2 archivos (source.png + project.json)
         console.log('📂 Cargando project.json...');
         const jsonResponse = await fetch(`output/${projectName}/project.json`);
         project = await jsonResponse.json();
         
         N = project.metadata.N;
         patches = project.patches;
         
         console.log(`✓ JSON cargado: ${patches.length} patches definidos`);
         
         // Cargar imagen original (una sola petición HTTP)
         console.log('🖼️ Cargando source.png...');
         const sourceImg = new Image();
         await new Promise((resolve, reject) => {
             sourceImg.onload = () => resolve();
             sourceImg.onerror = () => reject(new Error('Failed to load source.png'));
             sourceImg.src = `output/${projectName}/source.png`;
         });
         
         console.log(`✓ Imagen original cargada: ${sourceImg.width}x${sourceImg.height}`);
         
         // Extraer patches desde la imagen original usando canvas
         console.log('✂️ Extrayendo patches desde source.png...');
         patchImages = {};
         
         // Crear canvas temporal para extraer regiones
         const tempCanvas = document.createElement('canvas');
         const tempCtx = tempCanvas.getContext('2d');
         
         for (let i = 0; i < patches.length; i++) {
             const patch = patches[i];
             const [r, c] = patch.id;
             const [x0, y0, x1, y1] = patch.bbox;
             
             const patchWidth = x1 - x0;
             const patchHeight = y1 - y0;
             
             // Configurar canvas temporal
             tempCanvas.width = patchWidth;
             tempCanvas.height = patchHeight;
             
             // Extraer región desde la imagen original
             tempCtx.drawImage(
                 sourceImg,
                 x0, y0, patchWidth, patchHeight,  // source rect
                 0, 0, patchWidth, patchHeight     // dest rect
             );
             
             // Convertir a Image para usar en drawImage()
             const patchImg = new Image();
             patchImg.src = tempCanvas.toDataURL();
             patchImages[`${r}_${c}`] = patchImg;
             
             // Mostrar progreso cada 10%
             if (i % Math.floor(patches.length / 10) === 0) {
                 const progress = Math.round((i / patches.length) * 100);
                 ctx.fillStyle = '#0a0a1a';
                 ctx.fillRect(0, 0, canvas.width, canvas.height);
                 ctx.fillStyle = '#00ff88';
                 ctx.fillText(`EXTRACTING PATCHES... ${progress}%`, canvas.width / 2, canvas.height / 2);
             }
         }
         
         console.log(`✅ ${patches.length} patches extraídos (0 peticiones HTTP adicionales)`);
         console.log(`⚡ Velocidad: ~${patches.length}x más rápido que cargar archivos individuales`);
         
         // Reset view
         viewX = 0;
         viewY = 0;
         viewW = 1;
         viewH = 1;
         
         isViewDirty = true; // Marcar para el primer render
         
     } catch (error) {
         console.error('Error loading project:', error);
         alert(`Error al cargar proyecto: ${error.message}`);
     }
 }
 
 // ==================== NUEVA ARQUITECTURA DE RENDERIZADO ====================
 
 /**
  * Bucle principal de dibujado (60fps).
  * Es "tonto": solo dibuja si la vista está sucia o dibuja la vista cacheada.
  */
 function drawLoop() {
     requestAnimationFrame(drawLoop);
     
     if (!project) return; // No hay proyecto cargado
     
     if (isViewDirty && !isCalculating) {
         // Verificar si realmente necesitamos recalcular
         if (shouldRecalculate()) {
             startBatchCalculation();
         } else {
             // No es necesario recalcular, solo actualizar drawList con las coordenadas
             updateDrawListCoordinates();
             isViewDirty = false;
         }
     }
     
     drawCachedView(); // 2. Dibujar la vista (barato)
     
     // Dibujar indicador de progreso si estamos calculando
     if (isCalculating) {
         drawProgressIndicator();
     }
 }
 
 /**
  * Determina si necesitamos recalcular la subdivisión completa.
  * Solo recalcula cuando el nivel de LOD cambia significativamente.
  */
 function shouldRecalculate() {
     const currentZoom = 1 / viewW;
     const currentLODLevel = Math.floor(Math.log2(currentZoom));
     
     // Calcular la diferencia de nivel de LOD
     const lodDifference = Math.abs(currentLODLevel - lastLODLevel);
     
     // Calcular cuánto nos hemos movido
     const panDistance = Math.hypot(viewX - lastViewX, viewY - lastViewY);
     const panRatio = panDistance / viewW;
     
     // Recalcular si:
     // 1. El nivel de LOD cambió significativamente
     // 2. Nos movimos mucho (salimos del área cacheada)
     // 3. Es la primera vez
     const needsRecalc = 
         lastLODLevel === -1 || 
         lodDifference >= LOD_CHANGE_THRESHOLD ||
         panRatio > PAN_THRESHOLD;
     
     if (needsRecalc) {
         console.log(`🔄 Recalculando: LOD ${lastLODLevel} → ${currentLODLevel}, Pan: ${(panRatio * 100).toFixed(1)}%`);
     }
     
     return needsRecalc;
 }
 
 /**
  * Actualiza solo las coordenadas de la drawList existente sin recalcular subdivisiones.
  * Esto es MUY rápido y permite pan/zoom suaves sin recálculos.
  */
 function updateDrawListCoordinates() {
     if (drawList.length === 0) return;
     
     // Solo recalcular las coordenadas de pantalla para cada parche
     for (let i = 0; i < drawList.length; i++) {
         const item = drawList[i];
         const patch = item.patch;
         
         // Recalcular coordenadas normalizadas desde el patch
         // (esto requiere guardar las coordenadas normalizadas originales)
         if (!item.nx) continue; // Skip si no tiene coordenadas normalizadas
         
         const nx = item.nx;
         const ny = item.ny;
         const nw = item.nw;
         const nh = item.nh;
         
         // Recalcular coordenadas de pantalla
         item.sx = (nx - viewX) / viewW * CANVAS_SIZE;
         item.sy = (ny - viewY) / viewH * CANVAS_SIZE;
         item.sw = nw / viewW * CANVAS_SIZE;
         item.sh = nh / viewH * CANVAS_SIZE;
     }
     
     console.log(`✨ Actualización rápida: ${drawList.length} parches reposicionados`);
 }
 
 /**
  * Inicia el cálculo por lotes (batch processing)
  */
 function startBatchCalculation() {
     isCalculating = true;
     calculationProgress = 0;
     drawList = []; // Limpiar la lista
     batchQueue = [];
     batchIndex = 0;
     
     // Guardar el estado actual para la siguiente comparación
     const currentZoom = 1 / viewW;
     lastLODLevel = Math.floor(Math.log2(currentZoom));
     lastViewX = viewX;
     lastViewY = viewY;
     
     if (!project || patches.length === 0) {
         isCalculating = false;
         isViewDirty = false;
         return;
     }
     
     const normPatchW = 1.0 / N;
     const normPatchH = 1.0 / N;
     
     // Crear cola de trabajo: todos los parches base
     for (let r = 0; r < N; r++) {
         for (let c = 0; c < N; c++) {
             const nx = c * normPatchW;
             const ny = r * normPatchH;
             batchQueue.push({ r, c, nx, ny, nw: normPatchW, nh: normPatchH });
         }
     }
     
     console.log(`🚀 Iniciando cálculo por lotes: ${batchQueue.length} parches base`);
     
     // Iniciar el procesamiento
     processBatch();
 }
 
 /**
  * Procesa un batch de parches (se llama recursivamente con requestIdleCallback)
  */
 function processBatch() {
     if (batchIndex >= batchQueue.length) {
         // Terminamos
         finishCalculation();
         return;
     }
     
     // Procesar un batch de parches
     const endIndex = Math.min(batchIndex + BATCH_SIZE, batchQueue.length);
     
     for (let i = batchIndex; i < endIndex; i++) {
         const task = batchQueue[i];
         renderNodeCalculator(task.r, task.c, task.nx, task.ny, task.nw, task.nh);
     }
     
     batchIndex = endIndex;
     calculationProgress = Math.round((batchIndex / batchQueue.length) * 100);
     
     // Continuar en el próximo idle callback (o en el siguiente frame)
     if (window.requestIdleCallback) {
         requestIdleCallback(() => processBatch(), { timeout: 16 });
     } else {
         setTimeout(() => processBatch(), 0);
     }
 }
 
 /**
  * Finaliza el cálculo
  */
 function finishCalculation() {
     isCalculating = false;
     isViewDirty = false;
     calculationProgress = 100;
     
     console.log(`✅ Cálculo completado: ${drawList.length} parches en drawList`);
     
     updateInfo(); // Actualizar la UI
 }
 
 /**
  * Calcula la lista de parches visibles y la guarda en 'drawList'.
  * Esta es la función costosa que solo se ejecuta cuando la vista cambia.
  */
 function updateView() {
     // DEPRECATED: Reemplazada por startBatchCalculation()
     // Mantenida para compatibilidad
     startBatchCalculation();
 }
 
 /**
  * Dibuja el contenido del 'drawList' cacheado en el canvas.
  * Esta función es muy rápida.
  */
 function drawCachedView() {
     ctx.fillStyle = '#000';
     ctx.fillRect(0, 0, canvas.width, canvas.height);
     
     // 1. Aplicar efectos globales de GPU
     if (brightnessFactor !== 1.0) {
         ctx.filter = `brightness(${brightnessFactor * 100}%)`;
     }
     if (blendEnabled) {
         ctx.globalCompositeOperation = 'lighter'; // Blend Aditivo
     }
     
     // 2. Dibujar la lista de parches
     for (const item of drawList) {
         renderPatchOptimizado(item);
     }
     
     // 3. Resetear efectos de GPU
     if (brightnessFactor !== 1.0) {
         ctx.filter = 'none';
     }
     if (blendEnabled) {
         ctx.globalCompositeOperation = 'source-over';
     }
 }
 
 /**
  * Dibuja un indicador de progreso mientras se calcula
  */
 function drawProgressIndicator() {
     const barWidth = 200;
     const barHeight = 20;
     const x = (canvas.width - barWidth) / 2;
     const y = canvas.height - 50;
     
     // Fondo de la barra
     ctx.fillStyle = 'rgba(26, 26, 26, 0.8)'; // Fixed missing parenthesis
     ctx.fillRect(x - 5, y - 5, barWidth + 10, barHeight + 10);
     
     // Borde
     ctx.strokeStyle = '#ffffff';
     ctx.lineWidth = 2;
     ctx.strokeRect(x - 5, y - 5, barWidth + 10, barHeight + 10);
     
     // Barra de progreso
     const progress = calculationProgress / 100;
     ctx.fillStyle = '#ffffff'; // Changed to white for better visibility
     ctx.fillRect(x, y, barWidth * progress, barHeight);
     
     // Texto
     ctx.fillStyle = '#ffffff';
     ctx.font = '12px monospace';
     ctx.textAlign = 'center';
     ctx.fillText(`CALCULANDO... ${calculationProgress}%`, canvas.width / 2, y - 10);
 }
 
 /**
  * (NUEVO) Reemplaza a 'renderNode'.
  * Esta función es un *calculador*. No dibuja, solo
  * calcula la subdivisión y añade parches visibles al 'drawList'.
  */
 function renderNodeCalculator(r, c, nx, ny, nw, nh) {
     // Frustum culling
     if (nx + nw < viewX || nx > viewX + viewW ||
         ny + nh < viewY || ny > viewY + viewH) {
         return;
     }
     
     // Calculate screen coordinates
     const sx = (nx - viewX) / viewW * CANVAS_SIZE;
     const sy = (ny - viewY) / viewH * CANVAS_SIZE;
     const sw = nw / viewW * CANVAS_SIZE;
     const sh = nh / viewH * CANVAS_SIZE;
     
     const idx = r * N + c;
     if (idx >= patches.length) return;
     
     const patch = patches[idx];
     
     // LOD check
     if (sw < lodThreshold || sh < lodThreshold ||
         !patch.match || patch.match.k <= 1) {
         
         // CASO BASE: Añadir al drawList y parar recursión
         // Guardar también las coordenadas normalizadas para actualización rápida
         drawList.push({ 
             patch, 
             sx, sy, sw, sh,
             nx, ny, nw, nh  // ← NUEVO: coordenadas normalizadas
         });
         return;
     }
     
     // CASO RECURSIVO: Subdividir
     const match = patch.match;
     const k = match.k;
     const [topR, topC] = match.top;
     
     const childNW = nw / k;
     const childNH = nh / k;
     
     for (let rr = 0; rr < k; rr++) {
         for (let cc = 0; cc < k; cc++) {
             const childR = topR + rr;
             const childC = topC + cc;
             
             if (childR < 0 || childR >= N || childC < 0 || childC >= N) {
                 continue;
             }
             
             const childNX = nx + cc * childNW;
             const childNY = ny + rr * childNH;
             
             renderNodeCalculator(childR, childC, childNX, childNY, childNW, childNH);
         }
     }
 }
 
 /**
  * (NUEVO) Dibuja un parche en la pantalla con borde negro opcional.
  */
 function renderPatchOptimizado(item) {
     const { patch, sx, sy, sw, sh } = item;
     const [r, c] = patch.id;
     const img = patchImages[`${r}_${c}`];
     
     if (!img || !img.complete) return;
     
     try {
         // Si el blend está activado, reducir el área de dibujado para crear el borde
         if (blendEnabled && borderThickness > 0) {
             const halfBorder = borderThickness / 2;
             
             // Dibujar el parche con un margen interno
             ctx.drawImage(img, 
                 Math.round(sx + halfBorder), 
                 Math.round(sy + halfBorder), 
                 Math.round(sw - borderThickness), 
                 Math.round(sh - borderThickness)
             );
         } else {
             // Dibujado normal sin borde
             ctx.drawImage(img, 
                 Math.round(sx), 
                 Math.round(sy), 
                 Math.round(sw), 
                 Math.round(sh)
             );
         }
     } catch (error) {
         // ignorar errores de dibujado
     }
 }
 
 function updateInfo() {
     document.getElementById('zoom-level').textContent = (1 / viewW).toFixed(2) + 'x';
     document.getElementById('lod-value').textContent = lodThreshold + 'px';
     document.getElementById('brightness-value').textContent = brightnessFactor.toFixed(2) + 'x';
     document.getElementById('blend-status').textContent = blendEnabled ? `${borderThickness}px` : 'OFF';
 }
 
 // ==================== CONTROLS (Refactorizados) ====================
 // Los controles ahora solo actualizan el estado y marcan la vista como 'sucia'.
 
 function adjustBrightness(delta) {
     brightnessFactor = Math.max(0.3, Math.min(3.0, brightnessFactor + delta));
     isViewDirty = true;
 }
 
 function toggleBlend() {
     blendEnabled = !blendEnabled;
     isViewDirty = true;
 }
 
 function adjustBorderThickness(delta) {
     borderThickness = Math.max(0, Math.min(20, borderThickness + delta));
     console.log(`🎨 Grosor del borde: ${borderThickness}px`);
     isViewDirty = true;
 }
 
 function adjustLOD(delta) {
     lodThreshold = Math.max(1, Math.min(200, lodThreshold + delta));
     // ✨ NUEVO: Forzar recálculo cuando cambia LOD
     lastLODLevel = -1; // Invalidar el LOD cache para forzar recálculo
     isViewDirty = true;
 }
 
 // ==================== MOUSE EVENTS (Refactorizados) ====================
 canvas.addEventListener('mousedown', (e) => {
     isPanning = true;
     lastX = e.offsetX;
     lastY = e.offsetY;
 });
 
 canvas.addEventListener('mousemove', (e) => {
     if (!isPanning) return;
     
     const dx = e.offsetX - lastX;
     const dy = e.offsetY - lastY;
     lastX = e.offsetX;
     lastY = e.offsetY;
     
     const normDX = (dx / CANVAS_SIZE) * viewW;
     const normDY = (dy / CANVAS_SIZE) * viewH;
     
     viewX = Math.max(0, Math.min(1 - viewW, viewX - normDX));
     viewY = Math.max(0, Math.min(1 - viewH, viewY - normDY));
     
     isViewDirty = true; // Solo marcar, no renderizar
 });
 
 canvas.addEventListener('mouseup', () => {
     isPanning = false;
 });
 
 canvas.addEventListener('mouseleave', () => {
     isPanning = false;
 });
 
 canvas.addEventListener('wheel', (e) => {
     e.preventDefault();
     
     const zoomFactor = e.deltaY > 0 ? 1.05 : 0.95;
     const mx = e.offsetX / CANVAS_SIZE;
     const my = e.offsetY / CANVAS_SIZE;
     
     const imgMX = viewX + mx * viewW;
     const imgMY = viewY + my * viewH;
     
     const newW = viewW * zoomFactor;
     const newH = viewH * zoomFactor;
     
     viewX = imgMX - mx * newW;
     viewY = imgMY - my * newH;
     viewW = newW;
     viewH = newH;
     
     viewW = Math.max(0, Math.min(1, viewW));
     viewH = viewW;
     viewX = Math.max(0, Math.min(1 - viewW, viewX));
     viewY = Math.max(0, Math.min(1 - viewH, viewY));
     
     isViewDirty = true; // Solo marcar, no renderizar
 });
 
 // ==================== TOUCH EVENTS (Refactorizados) ====================
 canvas.addEventListener('touchstart', (e) => {
     e.preventDefault();
     
     if (e.touches.length === 1) {
         isPanning = true;
         const touch = e.touches[0];
         const rect = canvas.getBoundingClientRect();
         lastX = touch.clientX - rect.left;
         lastY = touch.clientY - rect.top;
     } else if (e.touches.length === 2) {
         const touch1 = e.touches[0];
         const touch2 = e.touches[1];
         touchStartDist = Math.hypot(
             touch2.clientX - touch1.clientX,
             touch2.clientY - touch1.clientY
         );
     }
 });
 
 canvas.addEventListener('touchmove', (e) => {
     e.preventDefault();
     
     if (e.touches.length === 1 && isPanning) {
         const touch = e.touches[0];
         const rect = canvas.getBoundingClientRect();
         const x = touch.clientX - rect.left;
         const y = touch.clientY - rect.top;
         
         const dx = x - lastX;
         const dy = y - lastY;
         lastX = x;
         lastY = y;
         
         const normDX = (dx / CANVAS_SIZE) * viewW;
         const normDY = (dy / CANVAS_SIZE) * viewH;
         
         viewX = Math.max(0, Math.min(1 - viewW, viewX - normDX));
         viewY = Math.max(0, Math.min(1 - viewH, viewY - normDY));
         
         isViewDirty = true;
     } else if (e.touches.length === 2) {
         const touch1 = e.touches[0];
         const touch2 = e.touches[1];
         const currentDist = Math.hypot(
             touch2.clientX - touch1.clientX,
             touch2.clientY - touch1.clientY
         );
         
         const zoomFactor = touchStartDist / currentDist;
         
         const centerX = (touch1.clientX + touch2.clientX) / 2;
         const centerY = (touch1.clientY + touch2.clientY) / 2;
         const rect = canvas.getBoundingClientRect();
         const mx = (centerX - rect.left) / CANVAS_SIZE;
         const my = (centerY - rect.top) / CANVAS_SIZE;
         
         const imgMX = viewX + mx * viewW;
         const imgMY = viewY + my * viewH;
         
         const newW = viewW * zoomFactor;
         const newH = viewH * zoomFactor;
         
         viewX = imgMX - mx * newW;
         viewY = imgMY - my * newH;
         viewW = newW;
         viewH = newH;
         
         viewW = Math.max(0.001, Math.min(1, viewW));
         viewH = viewW;
         viewX = Math.max(0, Math.min(1 - viewW, viewX));
         viewY = Math.max(0, Math.min(1 - viewH, viewY));
         
         touchStartDist = currentDist;
         isViewDirty = true;
     }
 });
 
 canvas.addEventListener('touchend', (e) => {
     if (e.touches.length === 0) {
         isPanning = false;
     }
 });
 
 // ==================== KEYBOARD SHORTCUTS (Sin cambios) ====================
 document.addEventListener('keydown', (e) => {
     switch(e.key.toLowerCase()) {
         case 'l':
             adjustBrightness(0.05);
             break;
         case 'k':
             adjustBrightness(-0.05);
             break;
         case 'b':
             toggleBlend();
             break;
         case 'arrowup':
         case 'u':
             adjustLOD(2);
             break;
         case 'arrowdown':
         case 'j':
             adjustLOD(-2);
             break;
         case 'arrowleft':
             adjustBorderThickness(-1);
             break;
         case 'arrowright':
             adjustBorderThickness(1);
             break;
     }
 });
 
 // ==================== INIT ====================
 drawLoop(); // Iniciar el bucle de renderizado
 loadProjects(); // Cargar la lista de proyectos