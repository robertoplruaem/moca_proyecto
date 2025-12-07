import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# Ajustamos la ruta relativa asumiendo que este script está en src/utils/
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent

FONTS_DIR = PROJECT_ROOT / "data" / "resources" / "fonts"
QUARANTINE_DIR = FONTS_DIR / "quarantine"  # Aquí moveremos las malas

def auditar_fuentes():
    if not FONTS_DIR.exists():
        print(f"❌ Error: No existe el directorio {FONTS_DIR}")
        return

    # Crear carpeta de cuarentena
    QUARANTINE_DIR.mkdir(exist_ok=True)

    # Listar todas las fuentes
    fuentes = list(FONTS_DIR.glob("*.ttf")) + list(FONTS_DIR.glob("*.otf"))
    print(f"🔍 Auditando {len(fuentes)} fuentes en busca de errores...")

    fuentes_malas = []
    fuentes_buenas = 0

    for font_path in tqdm(fuentes):
        es_valida = False
        razon = "Desconocida"

        try:
            # 1. Intento de carga
            font = ImageFont.truetype(str(font_path), size=64)
            
            # 2. Prueba de Tinta (Ink Test)
            # Intentamos dibujar caracteres clave
            test_chars = "A-0" 
            ancho_estimado = int(font.getlength(test_chars)) + 20
            
            # Crear lienzo temporal
            img = Image.new("1", (ancho_estimado, 100), 0) # Fondo negro
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), test_chars, font=font, fill=1) # Texto blanco
            
            bbox = img.getbbox()
            
            if bbox is None:
                razon = "Invisible (No dibuja nada)"
            elif (bbox[2] - bbox[0]) < 5 or (bbox[3] - bbox[1]) < 5:
                razon = "Dimensión minúscula (posible error de métrica)"
            else:
                # 3. Prueba del Guion (Crítico para tu dataset)
                # Si la fuente dibuja letras pero no tiene guion, también la marcamos
                # (Opcional: Si prefieres conservarlas, comenta este bloque else-if)
                if font.getbbox("-") is None:
                    razon = "Falta el carácter guion '-'"
                else:
                    es_valida = True

        except Exception as e:
            razon = f"Error de carga: {str(e)}"

        # --- VEREDICTO ---
        if es_valida:
            fuentes_buenas += 1
        else:
            fuentes_malas.append((font_path, razon))
            # Mover a cuarentena
            nombre_archivo = font_path.name
            destino = QUARANTINE_DIR / nombre_archivo
            try:
                shutil.move(str(font_path), str(destino))
            except Exception as e:
                print(f"   ⚠️ No se pudo mover {nombre_archivo}: {e}")

    # --- REPORTE FINAL ---
    print("\n" + "="*40)
    print(f"✅ Auditoría Finalizada")
    print("="*40)
    print(f"Fuentes Válidas:  {fuentes_buenas}")
    print(f"Fuentes Dañadas:  {len(fuentes_malas)}")
    
    if fuentes_malas:
        print(f"\n🗑️ Se movieron {len(fuentes_malas)} fuentes a: {QUARANTINE_DIR}")
        print("\nDetalle de errores:")
        for ruta, motivo in fuentes_malas[:10]: # Mostrar solo las primeras 10
            print(f"   • {ruta.name}: {motivo}")
        if len(fuentes_malas) > 10:
            print(f"   ... y {len(fuentes_malas)-10} más.")
    else:
        print("\n✨ ¡Tu colección de fuentes está limpia! No se encontraron errores.")

if __name__ == "__main__":
    auditar_fuentes()