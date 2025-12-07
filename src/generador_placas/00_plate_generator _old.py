import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import glob
from tqdm import tqdm
import math
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Determinamos la raíz del proyecto basándonos en la ubicación de este archivo.
# Ubicación: PROYECTO/src/generador_placas/generator.py
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent # subir a la raíz del proyecto

# Rutas dinámicas
DIR_FUENTES = str(PROJECT_ROOT / "data" / "resources" / "fonts")
DIR_PLANTILLAS = str(PROJECT_ROOT / "data" / "resources" / "templates")
# La salida irá a la carpeta de procesado intermedio
DIR_DATASET = str(PROJECT_ROOT / "data" / "processed" / "03_caracteres_sinteticos")

# --- Constantes Globales de la Placa ---
ANCHO_PLACA = 400
ALTO_PLACA = 200
UPSCALE_FACTOR = 2
ANCHO_TRABAJO = ANCHO_PLACA * UPSCALE_FACTOR
ALTO_TRABAJO = ALTO_PLACA * UPSCALE_FACTOR

# Crear directorios si no existen (seguridad)
os.makedirs(DIR_FUENTES, exist_ok=True)
os.makedirs(DIR_PLANTILLAS, exist_ok=True)
os.makedirs(DIR_DATASET, exist_ok=True)

# --- CARGA DE RECURSOS ---
# Buscar fuentes
FUENTES_DISPONIBLES = glob.glob(os.path.join(DIR_FUENTES, "**/*.ttf"), recursive=True)
FUENTES_DISPONIBLES.extend(glob.glob(os.path.join(DIR_FUENTES, "**/*.otf"), recursive=True))

if not FUENTES_DISPONIBLES:
    print(f"ADVERTENCIA: No se encontró ninguna fuente en '{DIR_FUENTES}'.")
    print("Por favor, coloca tus archivos .ttf/.otf en esa carpeta.")
else:
    print(f"Cargadas {len(FUENTES_DISPONIBLES)} fuentes.")

# Cargar plantillas en memoria (evitar I/O repetido)
PLANTILLAS_CACHE = {}
def cargar_plantillas_en_memoria():
    """Carga todas las plantillas en memoria al inicio"""
    global PLANTILLAS_CACHE
    lista_plantillas = glob.glob(os.path.join(DIR_PLANTILLAS, "*.jpg"))
    lista_plantillas.extend(glob.glob(os.path.join(DIR_PLANTILLAS, "*.png")))
    lista_plantillas.extend(glob.glob(os.path.join(DIR_PLANTILLAS, "*.jpeg")))
    
    if not lista_plantillas:
        print(f"ADVERTENCIA: No se encontraron plantillas en '{DIR_PLANTILLAS}'.")
        return []

    print(f"Cargando {len(lista_plantillas)} plantillas en memoria...")
    for ruta in lista_plantillas:
        try:
            img = Image.open(ruta)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            PLANTILLAS_CACHE[ruta] = img.copy()
            img.close()
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")
    
    return list(PLANTILLAS_CACHE.keys())

# --- OPTIMIZACIÓN: Cache de fuentes ---
FONT_CACHE = {}
def get_cached_font(font_path, size):
    key = (font_path, size)
    if key not in FONT_CACHE:
        try:
            FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        except Exception as e:
            # print(f"Error cargando fuente {font_path}: {e}")
            return None
    return FONT_CACHE[key]

# --- Funciones de utilidad ---
def get_random_color(tipo='fuerte'):
    if tipo == 'tenue':
        r, g, b = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        if r > 230 and g > 230 and b > 230: r -= 30
        return (r, g, b)
    else:
        r, g, b = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        if r < 50 and g < 50 and b < 50: g += 100
        return (r, g, b)

def generar_texto_placa():
    letras = string.ascii_uppercase
    numeros = string.digits
    formato = random.choice(['LLL-NNN-L', 'LLL-NN-NN', 'LNN-LLL', 'L-NNN-LLL', 'LLL-NNN'])
    
    if formato == 'LLL-NNN-L':
        placa = f"{''.join(random.choices(letras, k=3))}-{''.join(random.choices(numeros, k=3))}-{random.choice(letras)}"
    elif formato == 'LLL-NN-NN':
        placa = f"{''.join(random.choices(letras, k=3))}-{''.join(random.choices(numeros, k=2))}-{''.join(random.choices(numeros, k=2))}"
    elif formato == 'LNN-LLL':
        placa = f"{random.choice(letras)}{''.join(random.choices(numeros, k=2))}-{''.join(random.choices(letras, k=3))}"
    elif formato == 'L-NNN-LLL':
        placa = f"{random.choice(letras)}-{''.join(random.choices(numeros, k=3))}-{''.join(random.choices(letras, k=3))}"
    elif formato == 'LLL-NNN':
        placa = f"{''.join(random.choices(letras, k=3))}-{''.join(random.choices(numeros, k=3))}"   
    
    return placa

def get_pixel_width(fuente, texto):
    try:
        bbox = fuente.getbbox(texto)
        return bbox[2] - bbox[0]
    except Exception:
        return fuente.getlength(texto)

COLORED_TEXT_PROB = 0.2

def estampar_placa(ruta_plantilla, lista_fuentes, texto_placa):
    if ruta_plantilla not in PLANTILLAS_CACHE:
        return None, None, None
    
    img = PLANTILLAS_CACHE[ruta_plantilla].copy()
    img = img.resize((ANCHO_TRABAJO, ALTO_TRABAJO), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    ANCHO_IMG_LOCAL, ALTO_IMG_LOCAL = img.size
    
    size_placa = int(ALTO_IMG_LOCAL * 0.44)
    max_ancho_permitido = ANCHO_IMG_LOCAL * 0.85
    
    max_intentos = 10
    for intento in range(max_intentos):
        try:
            ruta_font_principal = random.choice(lista_fuentes)
            size_guion = int(size_placa * (0.31 / 0.44))
            
            fnt_placa = get_cached_font(ruta_font_principal, size_placa)
            if fnt_placa is None: continue
            
            # Intento de cargar arial para guiones, si no usa la misma
            fnt_guion_generico = fnt_placa 
            # Podrías agregar lógica para buscar arial específicamente si lo deseas
            
        except Exception:
            if intento == max_intentos - 1: return None, None, None
            continue
        
        ancho_total_calculado = 0
        espacio_char = int(ANCHO_IMG_LOCAL * 0.005)
        espacio_guion = int(ANCHO_IMG_LOCAL * 0.015)

        for char in texto_placa:
            font_temp = fnt_guion_generico if char == '-' else fnt_placa
            espacio_temp = espacio_guion if char == '-' else espacio_char
            ancho_total_calculado += get_pixel_width(font_temp, char) + espacio_temp
        
        if ancho_total_calculado <= max_ancho_permitido:
            break
        else:
            ratio = max_ancho_permitido / ancho_total_calculado
            size_placa = int(size_placa * ratio * 0.98)
            if size_placa < 10:
                if intento == max_intentos - 1: return None, None, None
                continue

    # use_colored_text = (random.random() < COLORED_TEXT_PROB)
    # color_texto_default = (255, 255, 255) if "_oscura" in ruta_plantilla else (0, 0, 0)
    # color_texto = get_random_color('fuerte') if use_colored_text else color_texto_default

    # --- LÓGICA DE CONTRASTE AUTOMÁTICO ---
    
    # 1. Muestrear el color de fondo en el centro de la placa
    # (Donde irá el texto)
    cx, cy = int(ANCHO_IMG_LOCAL / 2), int(ALTO_IMG_LOCAL / 2)
    
    # A veces la imagen puede tener canales alfa, aseguramos RGB
    try:
        color_fondo = img.getpixel((cx, cy))
        if isinstance(color_fondo, int): # Si es escala de grises
             color_fondo = (color_fondo, color_fondo, color_fondo)
        elif len(color_fondo) > 3: # Si tiene alpha (RGBA)
             color_fondo = color_fondo[:3]
    except Exception:
        color_fondo = (128, 128, 128) # Fallback gris neutro

    lum_fondo = calcular_luminancia(color_fondo)
    
    # Configuración
    UMBRAL_CONTRASTE = 110  # Diferencia mínima de brillo (0-255). 
                           # 80 es un buen balance. Si lo subes a 100 es más estricto.

    use_colored_text = (random.random() < COLORED_TEXT_PROB)

    if use_colored_text:
        # Intentar generar un color aleatorio que contraste
        color_texto = get_random_color('fuerte')
        lum_texto = calcular_luminancia(color_texto)
        
        # Si el contraste es pobre, forzamos Blanco o Negro según el fondo
        if abs(lum_fondo - lum_texto) < UMBRAL_CONTRASTE:
            if lum_fondo < 128: 
                color_texto = (255, 255, 255) # Fondo oscuro -> Texto blanco
            else: 
                color_texto = (0, 0, 0)       # Fondo claro -> Texto negro
    else:
        # Modo B/N estricto (Mejor para OCR)
        if lum_fondo < 128:
            color_texto = (255, 255, 255) # Blanco
        else:
            color_texto = (0, 0, 0)       # Negro
            
    # --- FIN LÓGICA CONTRASTE ---

    POS_Y_CENTRO = ALTO_IMG_LOCAL / 2
    pos_x_actual = (ANCHO_IMG_LOCAL / 2) - (ancho_total_calculado / 2)
    
    lista_bboxes_limpios = []
    
    # Referencia vertical con una letra estándar
    bbox_letra_control = fnt_placa.getbbox("X")
    if not bbox_letra_control: bbox_letra_control = (0,0,10,10) # Fallback

    control_y_centro_local = (bbox_letra_control[3] + bbox_letra_control[1]) / 2
    draw_y_letra_equivalente = POS_Y_CENTRO - control_y_centro_local
    CONTROL_Y1 = draw_y_letra_equivalente + bbox_letra_control[1]
    CONTROL_Y2 = draw_y_letra_equivalente + bbox_letra_control[3]

    mask_img = Image.new('L', (ANCHO_IMG_LOCAL, ALTO_IMG_LOCAL), 255)
    mask_draw = ImageDraw.Draw(mask_img)

    for char in texto_placa:
        if char == '-':
            font_actual = fnt_guion_generico
            espacio = espacio_guion
        else:
            font_actual = fnt_placa
            espacio = espacio_char

        bbox_glyph = font_actual.getbbox(char)
        if not bbox_glyph: # Caracter vacío (espacio)
             pos_x_actual += espacio * 3 # Dar un ancho al espacio
             continue

        char_ancho_pixels = bbox_glyph[2] - bbox_glyph[0]
        char_centro_y_local = (bbox_glyph[3] + bbox_glyph[1]) / 2
        draw_y = POS_Y_CENTRO - char_centro_y_local
        draw_x = pos_x_actual - bbox_glyph[0]
        
        draw.text((draw_x, draw_y), char, font=font_actual, fill=color_texto)
        mask_draw.text((draw_x, draw_y), char, font=font_actual, fill=0)
        
        if char == '-':
            abs_x1 = draw_x + bbox_glyph[0]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y1 = CONTROL_Y1
            abs_y2 = CONTROL_Y2
        else:
            abs_x1 = draw_x + bbox_glyph[0]
            abs_y1 = draw_y + bbox_glyph[1]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y2 = draw_y + bbox_glyph[3]

        char_bbox_data = {
            "char": char,
            "bbox_limpio": [abs_x1, abs_y1, abs_x2, abs_y2]
        }
        lista_bboxes_limpios.append(char_bbox_data)
        pos_x_actual += char_ancho_pixels + espacio

    return img, mask_img, lista_bboxes_limpios

def applying_augmentations_vectorized(img_pil, bboxes_limpios, mask_pil=None):
    # Configuración de Aumentación
    BLUR_MULTIPLIER_MIN, BLUR_MULTIPLIER_MAX = 1.5, 3
    NOISE_MULTIPLIER_MIN, NOISE_MULTIPLIER_MAX = 1.5, 3
    PROB_BLUR, PROB_NOISE, PROB_JPG = 1, 1, 1
    PROB_OCCLUSION, PROB_INVERT = 0.5, 0.4

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    alto, ancho = img_cv.shape[:2]

    mask_cv = None
    if mask_pil is not None:
        if mask_pil.mode != 'L': mask_pil = mask_pil.convert('L')
        mask_cv = np.array(mask_pil)

    if random.random() < PROB_INVERT:
        img_cv = cv2.bitwise_not(img_cv)
    
    if random.random() < 0.5:
        borderValue = (random.randint(80, 150), random.randint(80, 150), random.randint(80, 150))
    else:
        g = random.randint(50, 200)
        borderValue = (g, g, g)

    # Perspectiva
    pts_origen = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    MAX_ROTATION_Z = 30
    angulo = math.radians(random.uniform(-MAX_ROTATION_Z, MAX_ROTATION_Z))
    cos_a, sin_a = math.cos(angulo), math.sin(angulo)
    shear_factor = random.uniform(-0.6, 0.6) * 0.5
    escala = random.uniform(0.6, 1.0)
    cx, cy = ancho / 2, alto / 2
    
    pts_destino = []
    for (x, y) in pts_origen:
        x_scaled, y_scaled = (x - cx) * escala, (y - cy) * escala
        x_rot, y_rot = x_scaled * cos_a - y_scaled * sin_a, x_scaled * sin_a + y_scaled * cos_a
        x_shear, y_shear = x_rot + y_rot * shear_factor, y_rot
        pts_destino.append([x_shear + cx, y_shear + cy])
    
    matriz = cv2.getPerspectiveTransform(pts_origen, np.float32(pts_destino))

    img_rotated_cv = cv2.warpPerspective(img_cv, matriz, (ancho, alto), borderValue=borderValue)
    
    mask_rotated_cv = None
    if mask_cv is not None:
        mask_rotated_cv = cv2.warpPerspective(mask_cv, matriz, (ancho, alto), borderValue=255, flags=cv2.INTER_NEAREST)

    img_cv = img_rotated_cv.copy()

    # Oclusiones
    if random.random() < PROB_OCCLUSION:
        for _ in range(random.randint(1, 4)):
            cx_o, cy_o = random.randint(0, ancho), random.randint(0, alto)
            r_o = random.randint(2, 8)
            c_o = random.randint(20, 100)
            cv2.circle(img_cv, (cx_o, cy_o), r_o, (c_o, c_o, c_o), -1)

    # Brillo/Contraste
    if random.random() < 0.2:
        offset = int(random.uniform(-100, -50))
        con = random.uniform(0.6, 1.0)
    else:
        offset = int(random.uniform(-60, 60))
        con = random.uniform(0.7, 1.4)
    img_cv = cv2.convertScaleAbs(img_cv, alpha=con, beta=offset)
    
    # Blur
    if random.random() < PROB_BLUR:
        mult = random.uniform(BLUR_MULTIPLIER_MIN, BLUR_MULTIPLIER_MAX)
        k = int(random.choice([3, 5]) * mult)
        k = max(3, k if k % 2 != 0 else k + 1)
        img_cv = cv2.GaussianBlur(img_cv, (k, k), 0.8 * mult)

    # Ruido
    if random.random() < PROB_NOISE:
        mult = random.uniform(NOISE_MULTIPLIER_MIN, NOISE_MULTIPLIER_MAX)
        noise = np.random.normal(0, random.uniform(5, 25) * mult, img_cv.shape).astype(np.float32)
        img_cv = np.clip(img_cv.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPG
    if random.random() < PROB_JPG:
        _, buf = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 75)])
        img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Transformar BBoxes
    corners_placa = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]]).reshape(4, 1, 2)
    corners_placa_tf = cv2.perspectiveTransform(corners_placa, matriz)
    
    lista_bboxes_tf = []
    for item in bboxes_limpios:
        x1, y1, x2, y2 = item["bbox_limpio"]
        c_char = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(4, 1, 2)
        c_char_tf = cv2.perspectiveTransform(c_char, matriz)
        lista_bboxes_tf.append({"char": item["char"], "corners_tf": c_char_tf})
    
    (xp, yp, wp, hp) = cv2.boundingRect(np.int32(corners_placa_tf))
    
    padx, pady = int(wp * 0.20), int(hp * 0.20)
    xp, yp = max(0, xp - padx), max(0, yp - pady)
    wp = min(ancho - xp, wp + padx * 2)
    hp = min(alto - yp, hp + pady * 2)

    return img_cv, img_rotated_cv, mask_rotated_cv, (xp, yp, wp, hp), lista_bboxes_tf

# --- WORKER PARA MULTIPROCESSING ---
def init_worker(plantilla_paths, fuentes):
    global PLANTILLAS_CACHE, FUENTES_DISPONIBLES, FONT_CACHE
    FUENTES_DISPONIBLES = list(fuentes)
    FONT_CACHE = {}
    PLANTILLAS_CACHE = {}
    
    # Cargar plantillas en el proceso hijo
    for ruta in plantilla_paths:
        try:
            img = Image.open(ruta)
            if img.mode != 'RGB': img = img.convert('RGB')
            img_rs = img.resize((ANCHO_TRABAJO, ALTO_TRABAJO), Image.Resampling.LANCZOS)
            PLANTILLAS_CACHE[ruta] = img_rs.copy()
            img.close()
        except: pass

def generar_imagen_worker(args):
    try:
        idx, ruta_plantilla, seed = args
        random.seed(seed + idx)
        np.random.seed((seed + idx) & 0xFFFFFFFF)

        texto = generar_texto_placa()
        img_pil, mask_pil, bboxes = estampar_placa(ruta_plantilla, FUENTES_DISPONIBLES, texto)

        if img_pil is None: return None

        res_aug = applying_augmentations_vectorized(img_pil, bboxes, mask_pil)
        img_aug, img_clean, mask_aug, (xc, yc, wc, hc), bboxes_tf = res_aug

        # Recortes y Resizes
        def smart_crop_resize(image, method=cv2.INTER_AREA):
            if image is None: return None
            try:
                crop = image[yc : yc + hc, xc : xc + wc]
                if crop.size == 0: return None
                return cv2.resize(crop, (ANCHO_PLACA, ALTO_PLACA), interpolation=method)
            except: return None

        img_final = smart_crop_resize(img_aug)
        img_clean_final = smart_crop_resize(img_clean)
        mask_final = smart_crop_resize(mask_aug, cv2.INTER_NEAREST)

        if img_final is None: return None

        # JSON Bboxes
        scale_x = ANCHO_PLACA / max(1, wc)
        scale_y = ALTO_PLACA / max(1, hc)
        json_data = []
        
        for item in bboxes_tf:
            pts = item["corners_tf"] - [xc, yc]
            pts = pts * [scale_x, scale_y]
            (bx, by, bw, bh) = cv2.boundingRect(np.int32(pts))
            
            # Convertir a formato YOLO (x_center, y_center, w, h) normalizado
            cx = (bx + bw/2) / ANCHO_PLACA
            cy = (by + bh/2) / ALTO_PLACA
            nw = bw / ANCHO_PLACA
            nh = bh / ALTO_PLACA
            
            # Clamp 0-1
            cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
            nw, nh = max(0, min(1, nw)), max(0, min(1, nh))
            
            # Guardar caracter
            json_data.append({
                "char": item["char"],
                "yolo_box": [cx, cy, nw, nh]
            })

        # Codificar imágenes en memoria
        def encode_jpg(img):
            _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(80, 95)])
            return buf.tobytes()
        
        return {
            'nombre': texto,
            'img_bytes': encode_jpg(img_final),
            'json_data': json_data
        }

    except Exception as e:
        return {'error': str(e)}

# --- FUNCIÓN PRINCIPAL PARA LLAMAR DESDE NOTEBOOK ---
def generar_dataset(cantidad, num_workers=None):
    """
    Genera el dataset de placas sintéticas.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
        
    # 1. Validar recursos
    plantillas = glob.glob(os.path.join(DIR_PLANTILLAS, "*.jpg")) + \
                 glob.glob(os.path.join(DIR_PLANTILLAS, "*.png"))
                 
    fuentes = glob.glob(os.path.join(DIR_FUENTES, "*.ttf")) + \
              glob.glob(os.path.join(DIR_FUENTES, "*.otf"))
              
    if not plantillas:
        raise FileNotFoundError(f"No se encontraron plantillas en: {DIR_PLANTILLAS}")
    if not fuentes:
        raise FileNotFoundError(f"No se encontraron fuentes en: {DIR_FUENTES}")

    print(f"Iniciando generación de {cantidad} placas con {num_workers} procesos...")
    
    # 2. Preparar argumentos
    args = []
    for i in range(cantidad):
        args.append((i, random.choice(plantillas), random.randint(0, 1000000)))

    # 3. Ejecutar Pool
    generados = 0
    # Usamos directorio de imágenes y etiquetas
    out_img_dir = os.path.join(DIR_DATASET, "images")
    out_lbl_dir = os.path.join(DIR_DATASET, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Diccionario para mapear caracteres a IDs (debe coincidir con tu data.yaml)
    # A-Z (0-25), 0-9 (26-35) - (36) 
    chars = string.ascii_uppercase + string.digits + "-" 
    char_to_id = {c: i for i, c in enumerate(chars)}

    with Pool(processes=num_workers, initializer=init_worker, initargs=(plantillas, fuentes)) as pool:
        for res in tqdm(pool.imap_unordered(generar_imagen_worker, args), total=cantidad):
            if res is None or 'error' in res: continue
            
            # Guardar Imagen
            name = f"{res['nombre']}_{random.randint(1000,9999)}"
            with open(os.path.join(out_img_dir, f"{name}.jpg"), 'wb') as f:
                f.write(res['img_bytes'])
                
            # Guardar Label YOLO .txt
            with open(os.path.join(out_lbl_dir, f"{name}.txt"), 'w') as f:
                for item in res['json_data']:
                    char = item['char']
                    if char in char_to_id:
                        cid = char_to_id[char]
                        box = item['yolo_box']
                        f.write(f"{cid} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            
            generados += 1
            
    print(f"Generación terminada. {generados} placas creadas en: {DIR_DATASET}")

def calcular_luminancia(color):
    """
    Calcula el brillo percibido de un color (RGB).
    Retorna un valor entre 0 (negro) y 255 (blanco).
    """
    r, g, b = color
    return 0.299 * r + 0.587 * g + 0.114 * b

if __name__ == '__main__':
    # Test simple si se ejecuta directo
    # generar_dataset(1000, num_workers=4)
    generar_dataset(1000, num_workers=6)