
import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Intentar importar módulos del generador
try:
    from generator import (
        estampar_placa, 
        generar_texto_placa, 
        apply_transformations,
        DIR_FUENTES, 
        DIR_PLANTILLAS,
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
    """Carga la plantilla en el caché global manejando transparencia."""
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
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return False

def mostrar_pasos_transformacion():
    print("🔍 Iniciando diagnóstico visual paso a paso...")
    
    if not PLANTILLAS_DISPONIBLES or not FUENTES_DISPONIBLES:
        print("Faltan recursos (fuentes/plantillas).")
        return

    # 1. Selección
    plantilla = random.choice(PLANTILLAS_DISPONIBLES)
    fuentes = FUENTES_DISPONIBLES 
    texto = generar_texto_placa()
    
    print(f"📄 Plantilla: {Path(plantilla).name}")
    print(f"🔤 Texto: {texto}")

    # Cargar en caché manual
    if not cargar_plantilla_manual(plantilla): return

    # 2. Generar Base
    img_pil, mask_pil, bboxes, err = estampar_placa(plantilla, fuentes, texto)
    
    if img_pil is None:
        print(f"Error al estampar: {err}")
        return

    # 3. Preparar visualización
    img_base_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 4. Transformar
    img_final, json_data = apply_transformations(img_pil, bboxes, mask_pil)
    img_final_resized = cv2.resize(img_final, (ANCHO_PLACA, ALTO_PLACA))

    # --- GENERAR GRÁFICA ---
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Limpia
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_base_cv, cv2.COLOR_BGR2RGB))
    plt.title("1. Estampado Limpio")
    plt.axis('off')
    
    # Plot 2: Máscara
    plt.subplot(1, 3, 2)
    plt.imshow(mask_pil, cmap='gray')
    plt.title("2. Máscara de Texto")
    plt.axis('off')
    
    # Plot 3: Resultado Final
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_final_resized, cv2.COLOR_BGR2RGB))
    plt.title("3. Final (Transformado)")
    plt.axis('off')
    
    # Dibujar cajas sobre el resultado final
    h, w = img_final_resized.shape[:2]
    for item in json_data:
        cx, cy, nw, nh = item['yolo_box']
        x1 = int((cx - nw/2) * w)
        y1 = int((cy - nh/2) * h)
        x2 = int((cx + nw/2) * w)
        y2 = int((cy + nh/2) * h)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.tight_layout()
    
    # --- GUARDAR EN DISCO (SOLUCIÓN) ---
    output_file = "pipeline_debug.png"
    plt.savefig(output_file, dpi=150)
    print(f"\n¡ÉXITO! Imagen guardada en: {os.path.abspath(output_file)}")
    
    # Intentar mostrar ventana (si el entorno lo permite)
    try:
        print("   Intentando abrir ventana gráfica...")
        plt.show()
    except Exception:
        print("No se pudo abrir la ventana (entorno sin GUI). Revisa el archivo PNG.")

if __name__ == "__main__":
    mostrar_pasos_transformacion()