import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Importar constantes y variables
try:
    from generator import (
        estampar_placa, 
        generar_texto_placa,
        DIR_PLANTILLAS, 
        DIR_FUENTES,
        FUENTES_DISPONIBLES, 
        PLANTILLAS_DISPONIBLES,
        PLANTILLAS_CACHE,
        ANCHO_PLACA, 
        ALTO_PLACA
    )
except ImportError:
    print("Error: No se pudo importar 'generator.py'.")
    sys.exit(1)

def cargar_plantilla_manual(ruta_plantilla):
    """Carga plantilla en caché manejando transparencia."""
    try:
        with Image.open(ruta_plantilla) as img:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert('RGB')
            PLANTILLAS_CACHE[str(ruta_plantilla)] = img.copy()
            return True
    except: return False

def visualizar_pipeline_detallado():
    print("Generando desglose paso a paso...")
    
    if not PLANTILLAS_DISPONIBLES:
        print("Faltan plantillas.")
        return

    # 1. Configuración Inicial
    plantilla = random.choice(PLANTILLAS_DISPONIBLES)
    fuentes = FUENTES_DISPONIBLES 
    texto = generar_texto_placa()
    
    if not cargar_plantilla_manual(plantilla): return

    # 2. Generar Base (Estampado)
    img_pil, mask_pil, bboxes, err = estampar_placa(plantilla, fuentes, texto)
    if img_pil is None:
        print(f"Error: {err}")
        return

    # --- INICIO DEL DESGLOSE DEL PIPELINE ---
    pasos_visuales = []
    titulos = []

    # PASO 0: Original
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    pasos_visuales.append(img.copy())
    titulos.append("1. Original (Estampado)")

    # PASO 1: Geometría
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
    
    # Máscara para recorte futuro
    mask_full = np.ones((h, w), dtype=np.uint8) * 255
    mask_tf = cv2.warpPerspective(mask_full, M, (w,h), borderValue=0)

    pasos_visuales.append(img.copy())
    titulos.append("2. Geometría (Perspectiva)")

    # PASO 2: Inversión (Forzamos 50% para ver si ocurre)
    es_negativo = False
    if random.random() < 0.5: # Subimos probabilidad solo para visualización
        img = cv2.bitwise_not(img)
        es_negativo = True
    
    pasos_visuales.append(img.copy())
    titulos.append(f"3. Inversión: {'SÍ' if es_negativo else 'NO'}")

    # PASO 3: Tono y Color
    b_s = random.uniform(0.9, 1.1)
    r_s = random.uniform(0.9, 1.1)
    B, G, R = cv2.split(img)
    B = cv2.multiply(B, b_s); R = cv2.multiply(R, r_s)
    img = cv2.merge([B, G, R])
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # PASO 4: Iluminación
    brillo = random.randint(-90, 25)
    contraste = random.uniform(0.8, 1.4)
    img = cv2.convertScaleAbs(img, alpha=contraste, beta=brillo)

    pasos_visuales.append(img.copy())
    titulos.append(f"4. Luz/Color (B:{brillo})")

    # PASO 5: Degradación
    k = random.choice([3, 5])
    img = cv2.GaussianBlur(img, (k, k), 0)
    noise = np.random.normal(0, random.randint(10, 30), img.shape).astype(np.int16)
    img = cv2.add(img.astype(np.int16), noise)
    img = np.clip(img, 0, 255).astype(np.uint8)

    pasos_visuales.append(img.copy())
    titulos.append("5. Degradación (Blur+Ruido)")

    # PASO 6: Recorte Final
    nz = cv2.findNonZero(mask_tf)
    if nz is not None:
        x, y, wn, hn = cv2.boundingRect(nz)
        img = img[y:y+hn, x:x+wn]
    
    # Resize final
    img = cv2.resize(img, (ANCHO_PLACA, ALTO_PLACA))
    
    pasos_visuales.append(img.copy())
    titulos.append("6. Salida Final (Recorte)")

    # --- GRAFICAR ---
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8))
    fig.suptitle(f'Pipeline Detallado: {texto} | Plantilla: {Path(plantilla).name}', fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(pasos_visuales):
            # Convertir BGR a RGB para matplotlib
            img_rgb = cv2.cvtColor(pasos_visuales[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(titulos[i], fontsize=11, fontweight='bold')
            
            # Dibujar caja roja en la final
            if i == 5: # Último paso
                h_f, w_f = img_rgb.shape[:2]
                border_style = dict(linewidth=2, edgecolor='red', facecolor='none')
                rect = plt.Rectangle((5, 5), w_f-10, h_f-10, **border_style)
                ax.add_patch(rect)
        ax.axis('off')

    plt.tight_layout()
    
    # Guardar
    out_path = "pipeline_steps.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nImagen guardada en: {os.path.abspath(out_path)}")
    
    try: plt.show()
    except: pass

if __name__ == "__main__":
    visualizar_pipeline_detallado()