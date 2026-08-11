import os
import random
import string
import json
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageStat

# --- 1. CONFIGURACIÓN DE RUTAS ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent

DIR_FUENTES = PROJECT_ROOT / "data" / "resources" / "fonts"
DIR_PLANTILLAS = PROJECT_ROOT / "data" / "resources" / "templates"
DIR_DATASET = PROJECT_ROOT / "data" / "processed" / "03_caracteres_sinteticos"

os.makedirs(DIR_FUENTES, exist_ok=True)
os.makedirs(DIR_PLANTILLAS, exist_ok=True)
os.makedirs(DIR_DATASET, exist_ok=True)

# --- 2. CONSTANTES ---
ANCHO_PLACA = 400
ALTO_PLACA = 200
UPSCALE_FACTOR = 2
ANCHO_TRABAJO = ANCHO_PLACA * UPSCALE_FACTOR
ALTO_TRABAJO = ALTO_PLACA * UPSCALE_FACTOR
PATH_ARIAL = DIR_FUENTES / "arial.ttf"

# --- 3. RECURSOS GLOBAL ---
FUENTES_DISPONIBLES = [str(p) for p in DIR_FUENTES.glob("*.*") if p.suffix.lower() in ['.ttf', '.otf']]
PLANTILLAS_DISPONIBLES = [str(p) for p in DIR_PLANTILLAS.glob("*.*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]

FONT_CACHE = {}
PLANTILLAS_CACHE = {}

# --- 4. FUNCIONES AUXILIARES ---

def get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in FONT_CACHE:
        try:
            FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        except Exception: return None
    return FONT_CACHE[key]

def get_pixel_width(fuente, texto):
    try:
        bbox = fuente.getbbox(texto)
        if bbox: return bbox[2] - bbox[0]
        return fuente.getlength(texto)
    except: return 0

def generar_texto_placa():
    #letras = string.ascii_uppercase
    #numeros = string.digits
    alfanumericos = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
    
    try:
        # Extraemos caracteres al azar de la MISMA bolsa para cualquier posición.
        # Esto garantiza que una 'Z' tenga exactamente la misma probabilidad de 
        # aparecer que un '9', balanceando el dataset orgánicamente.
        formato = random.choice(['FORMATO_7', 'FORMATO_6_A', 'FORMATO_6_B'])
        
        if formato == 'FORMATO_7':
            # Ej: XXX-XX-XX (7 caracteres + 2 guiones)
            placa = f"{''.join(random.choices(alfanumericos, k=3))}-{''.join(random.choices(alfanumericos, k=2))}-{''.join(random.choices(alfanumericos, k=2))}"
        elif formato == 'FORMATO_6_A':
            # Ej: X-XXX-XXX (7 caracteres + 2 guiones)
            placa = f"{random.choice(alfanumericos)}-{''.join(random.choices(alfanumericos, k=3))}-{''.join(random.choices(alfanumericos, k=3))}"
        elif formato == 'FORMATO_6_B':
            # Ej: XXX-XXX (6 caracteres + 1 guion)
            placa = f"{''.join(random.choices(alfanumericos, k=3))}-{''.join(random.choices(alfanumericos, k=3))}"
        else:
            placa = "ABC-123"
    except:
        placa = "ABC-123"
        
    if not placa or len(placa.strip()) == 0: return "ABC-123"
    return placa

def calcular_luminancia(color):
    """Retorna luminancia 0-255."""
    if hasattr(color, '__len__'): 
        return 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
    return float(color)

def find_safe_hyphen_font(lista_fuentes, size, fallback_absoluto_path):
    """Busca una fuente donante que tenga guion."""
    for _ in range(5):
        ruta = random.choice(lista_fuentes)
        font = get_cached_font(ruta, size)
        if font and font.getbbox("-") is not None:
            return font
    if os.path.exists(str(fallback_absoluto_path)):
        font = get_cached_font(str(fallback_absoluto_path), size)
        if font and font.getbbox("-") is not None:
            return font
    return None

# --- 5. LÓGICA DE ESTAMPADO ---

def estampar_placa(ruta_plantilla, lista_fuentes, texto_placa):
    if ruta_plantilla not in PLANTILLAS_CACHE:
        return None, None, None, f"No cache: {ruta_plantilla}"
    
    img = PLANTILLAS_CACHE[ruta_plantilla].copy()
    img = img.resize((ANCHO_TRABAJO, ALTO_TRABAJO), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    ANCHO_IMG, ALTO_IMG = img.size
    
    # 1. Configuración de Fuente
    size_placa = int(ALTO_IMG * 0.44)
    max_ancho = ANCHO_IMG * 0.85
    fnt_placa = None
    
    # A. Selección de Fuente
    for _ in range(15):
        ruta_font = random.choice(lista_fuentes)
        fnt_placa = get_cached_font(ruta_font, size_placa)
        
        if fnt_placa is None: continue
        test_img = Image.new('1', (50, 50), 0)
        ImageDraw.Draw(test_img).text((10, 10), "A0", font=fnt_placa, fill=1)
        if test_img.getbbox() is None: continue 

        # B. Fallback Guion
        fnt_guion = fnt_placa
        if fnt_placa.getbbox("-") is None:
            fnt_guion = find_safe_hyphen_font(lista_fuentes, size_placa, PATH_ARIAL)
            if fnt_guion is None: fnt_guion = fnt_placa

        # C. Medición
        ancho_total = 0
        espacio = int(ANCHO_IMG * 0.005)
        
        for char in texto_placa:
            font_actual = fnt_guion if char == '-' else fnt_placa
            w = get_pixel_width(font_actual, char)
            if w == 0: w = int(size_placa * 0.3)
            ancho_total += w + espacio
            
        if ancho_total <= max_ancho:
            break
        else:
            size_placa = int(size_placa * 0.85)

    if fnt_placa is None: return None, None, None, "No fuente valida"

    # D. Contraste (Franja Maestra)
    try:
        y1, y2 = int(ALTO_IMG * 0.40), int(ALTO_IMG * 0.60)
        x1, x2 = int(ANCHO_IMG * 0.05), int(ANCHO_IMG * 0.95)
        franja = img.crop((x1, y1, x2, y2))
        stat = ImageStat.Stat(franja)
        lum_fondo = calcular_luminancia(stat.mean[:3])
    except: lum_fondo = 140

    if lum_fondo > 110: 
        color_texto = (0, 0, 0) # Negro Puro
    else: 
        color_texto = (255, 255, 255) # Blanco Puro

    # E. Dibujado con Stroke
    grosor_borde = max(1, int(size_placa * 0.015)) 
    
    pos_x = (ANCHO_IMG - ancho_total) / 2
    pos_y = ALTO_IMG / 2
    
    mask = Image.new('L', (ANCHO_IMG, ALTO_IMG), 0)
    mask_draw = ImageDraw.Draw(mask)
    bboxes = []
    
    for char in texto_placa:
        font = fnt_guion if char == '-' else fnt_placa
        
        bbox = font.getbbox(char)
        if not bbox: bbox = (0,0,1,1)
        
        cw = bbox[2]-bbox[0]
        ch = bbox[3]-bbox[1]
        
        dy = pos_y - (ch/2) - bbox[1]
        dx = pos_x
        
        draw.text((dx, dy), char, font=font, fill=color_texto, 
                  stroke_width=grosor_borde, stroke_fill=color_texto)
        mask_draw.text((dx, dy), char, font=font, fill=255, 
                       stroke_width=grosor_borde, stroke_fill=255)
        
        bboxes.append({
            "char": char, 
            "bbox_limpio": [
                dx + bbox[0] - grosor_borde, 
                dy + bbox[1] - grosor_borde, 
                dx + bbox[2] + grosor_borde, 
                dy + bbox[3] + grosor_borde
            ]
        })
        pos_x += cw + int(ANCHO_IMG*0.005)

    if mask.getbbox() is None:
        return None, None, None, "Imagen vacía post-render"

    return img, mask, bboxes, None

# --- 6. TRANSFORMACIONES (CON INVERSIÓN) ---

def apply_transformations(img_pil, bboxes, mask_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # 1. GEOMETRÍA
    pts_src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    limit = 0.22 
    dx, dy = w * limit, h * limit
    
    pts_dst = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), h - random.uniform(0, dy)],
        [random.uniform(0, dx), h - random.uniform(0, dy)]
    ])
    
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img = cv2.warpPerspective(img, M, (w,h), borderValue=(114,114,114))
    
    mask_full = np.ones((h, w), dtype=np.uint8) * 255
    mask_plate_tf = cv2.warpPerspective(mask_full, M, (w,h), borderValue=0)
    
    # 2. INVERSIÓN (NEGATIVO) - NUEVO!
    # Invierte los colores (Blanco <-> Negro) para simular placas oscuras/nocturnas.
    # Se hace ANTES del ruido para que el ruido afecte igual.
    if random.random() < 0.25: # 25% de las imágenes serán negativas
        img = cv2.bitwise_not(img)

    # 3. TONO
    if random.random() < 0.7:
        b_s = random.uniform(0.9, 1.1)
        r_s = random.uniform(0.9, 1.1)
        B, G, R = cv2.split(img)
        B = cv2.multiply(B, b_s)
        R = cv2.multiply(R, r_s)
        img = cv2.merge([B, G, R])
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. ILUMINACIÓN
    if random.random() < 0.8:
        brightness = random.randint(-90, 25) 
        contrast = random.uniform(0.8, 1.4)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # 5. DEGRADACIÓN
    if random.random() < 0.6:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
    if random.random() < 0.5:
        noise = np.random.normal(0, random.randint(10, 30), img.shape).astype(np.int16)
        img = cv2.add(img.astype(np.int16), noise)
        img = np.clip(img, 0, 255).astype(np.uint8)

    # 6. RECORTE
    nz = cv2.findNonZero(mask_plate_tf)
    if nz is not None:
        x, y, wn, hn = cv2.boundingRect(nz)
    else:
        x, y, wn, hn = 0, 0, w, h
        
    img_fin = img[y:y+hn, x:x+wn]
    
    json_data = []
    for item in bboxes:
        bx1, by1, bx2, by2 = item['bbox_limpio']
        pts = np.float32([[[bx1,by1]],[[bx2,by1]],[[bx2,by2]],[[bx1,by2]]])
        t_pts = cv2.perspectiveTransform(pts, M)
        
        t_pts[:,:,0] -= x
        t_pts[:,:,1] -= y
        
        xmin, ymin = np.min(t_pts[:,:,0]), np.min(t_pts[:,:,1])
        xmax, ymax = np.max(t_pts[:,:,0]), np.max(t_pts[:,:,1])
        
        fh, fw = img_fin.shape[:2]
        if fw < 1 or fh < 1: continue
        
        json_data.append({
            "char": item['char'],
            "yolo_box": [((xmin+xmax)/2)/fw, ((ymin+ymax)/2)/fh, (xmax-xmin)/fw, (ymax-ymin)/fh]
        })
        
    return img_fin, json_data

# --- 7. WORKER ---

def init_worker(plantilla_paths, fuentes_paths):
    global PLANTILLAS_CACHE, FUENTES_DISPONIBLES, FONT_CACHE
    FONT_CACHE = {}
    PLANTILLAS_CACHE = {}
    FUENTES_DISPONIBLES = fuentes_paths
    
    for p in plantilla_paths:
        try:
            with Image.open(p) as img:
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGBA')
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                else:
                    img = img.convert('RGB')
                PLANTILLAS_CACHE[str(p)] = img.copy()
        except: pass

def worker_task(idx):
    try:
        random.seed(idx)
        np.random.seed(idx & 0xFFFFFFFF)
        
        if not PLANTILLAS_CACHE: return {'error': 'Cache vacio'}
            
        plantilla = random.choice(list(PLANTILLAS_CACHE.keys()))
        texto = generar_texto_placa()
        
        img_pil, mask_pil, bboxes, err = estampar_placa(plantilla, FUENTES_DISPONIBLES, texto)
        if img_pil is None: return {'error': f"Estampado: {err}"}
            
        img_cv, json_data = apply_transformations(img_pil, bboxes, mask_pil)
        
        img_cv = cv2.resize(img_cv, (ANCHO_PLACA, ALTO_PLACA))
        _, buf = cv2.imencode('.jpg', img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        return {'texto': texto, 'img': buf.tobytes(), 'data': json_data}
        
    except Exception as e:
        return {'error': str(e)}

# --- 8. MAIN ---

def generar_dataset(cantidad=2000):
    if not FUENTES_DISPONIBLES:
        print(f"ERROR: Sin fuentes en {DIR_FUENTES}")
        return
    if not PLANTILLAS_DISPONIBLES:
        print(f"ERROR: Sin plantillas en {DIR_PLANTILLAS}")
        return

    print(f"Generando {cantidad} placas (Con Inversión Negativa)...")
    
    img_dir = DIR_DATASET / "images"
    lbl_dir = DIR_DATASET / "labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    #chars = string.ascii_uppercase + string.digits + "-"
    chars = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ-"
    char_to_id = {c: i for i, c in enumerate(chars)}
    
    tasks = list(range(cantidad))
    exitos = 0
    errores_log = []
    
    with Pool(processes=max(1, cpu_count()-1), initializer=init_worker, initargs=(PLANTILLAS_DISPONIBLES, FUENTES_DISPONIBLES)) as pool:
        for res in tqdm(pool.imap_unordered(worker_task, tasks), total=cantidad):
            if not res: continue
            if 'error' in res:
                errores_log.append(res['error'])
                continue
                
            base = f"{res['texto']}_{random.randint(1000,9999)}"
            with open(img_dir / f"{base}.jpg", 'wb') as f:
                f.write(res['img'])
                
            lines = []
            for item in res['data']:
                if item['char'] in char_to_id:
                    cid = char_to_id[item['char']]
                    box = item['yolo_box']
                    lines.append(f"{cid} {box[0]:.5f} {box[1]:.5f} {box[2]:.5f} {box[3]:.5f}")
                    
            with open(lbl_dir / f"{base}.txt", 'w') as f:
                f.write('\n'.join(lines))
            exitos += 1

    print(f"\nTerminado. Exitos: {exitos} | Fallos: {len(errores_log)}")
    if errores_log: print("Errores:", errores_log[:3])

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    generar_dataset(cantidad=30000)