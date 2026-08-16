# Archivo: app/main_pipeline.py

import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# --- 1. IMPORTAR TU ARQUITECTURA ---
# Asegúrate de que el archivo y la clase se llamen exactamente así
from customOCR_CNN import CustomOCR_CNN

# --- 2. CONFIGURACIÓN DE GPU ---
# Esto asegura que la RTX 4070 Ti se utilice al máximo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Procesador PyTorch configurado en: {device}")

# --- 3. DICCIONARIO / ALFABETO ---
# Aquí asumo los 10 números, 26 letras y el guion (-). 
# IMPORTANTE: Ajusta el orden si en tu .ipynb original era diferente.
# ALFABETO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALFABETO = "-0123456789ABCDEFGHJKLMNPRSTUVWXYZ"

# --- 4. RUTAS A TUS MODELOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_AUTOS = os.path.join(BASE_DIR, '../production_weights/01_autos_best.pt')
RUTA_PLACAS = os.path.join(BASE_DIR, '../production_weights/02_placas_best.pt')
RUTA_OCR = os.path.join(BASE_DIR, '../production_weights/03_caracteres_best.pth')
RUTA_YOLO_CLS = os.path.join(BASE_DIR, '../production_weights/03_yolo_cls_best.pt')
detector_cls = YOLO(RUTA_YOLO_CLS)

# --- 5. INICIALIZAR MODELOS ---
print("Cargando modelos YOLO y PyTorch en memoria...")

# A. Cargar YOLO
try:
    detector_autos = YOLO(RUTA_AUTOS)
    detector_placas = YOLO(RUTA_PLACAS)
    print("Modelos YOLO listos.")
except Exception as e:
    print(f"Error cargando YOLO: {e}")

# B. Cargar tu CNN Personalizada
try:
    print("Cargando arquitectura y pesos de la CNN...")
    # Instanciamos con 33 clases, según el tensor que detectamos
    modelo_ocr = CustomOCR_CNN(num_classes=len(ALFABETO)) 
    
    # Cargar los pesos y enviarlo a la tarjeta de video
    pesos = torch.load(RUTA_OCR, map_location=device, weights_only=True)
    modelo_ocr.load_state_dict(pesos)
    modelo_ocr.to(device)
    modelo_ocr.eval() # Modo evaluación (importante)
    
    print("Red Neuronal CNN OCR lista y cargada en GPU.")
except Exception as e:
    print(f"Error crítico cargando la CNN: {e}")


# --- 6. FUNCIONES DE PROCESAMIENTO ---

def corregir_normativa_mexicana(caracter):
    """Mapea caracteres prohibidos (I, Ñ, O, Q) a sus pares legales."""
    mapeo = {'I': '1', 'Ñ': 'N', 'O': '0', 'Q': '0'}
    return mapeo.get(caracter, caracter)

def rectificar_perspectiva_placa(img_placa):
    """Detecta el contorno de la placa y aplica una transformación de perspectiva."""
    gris = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    
    # Suavizado para ignorar detalles internos (letras) y buscar solo los bordes
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(blur, 50, 150)
    
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        # Tomar el contorno más grande (la placa física)
        c_max = max(contornos, key=cv2.contourArea)
        perimetro = cv2.arcLength(c_max, True)
        aproximacion = cv2.approxPolyDP(c_max, 0.05 * perimetro, True)
        
        # Si logramos detectar 4 esquinas, aplanamos la imagen
        if len(aproximacion) == 4:
            pts = aproximacion.reshape(4, 2)
            
            # Ordenar las esquinas: Sup-Izq, Sup-Der, Inf-Der, Inf-Izq
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            
            # Calcular ancho y alto máximo de la nueva imagen plana
            ancho_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            ancho_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_ancho = max(int(ancho_a), int(ancho_b))
            
            alto_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            alto_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_alto = max(int(alto_a), int(alto_b))
            
            dst = np.array([
                [0, 0],
                [max_ancho - 1, 0],
                [max_ancho - 1, max_alto - 1],
                [0, max_alto - 1]
            ], dtype="float32")
            
            # Aplicar la matriz de transformación
            matriz = cv2.getPerspectiveTransform(rect, dst)
            placa_plana = cv2.warpPerspective(img_placa, matriz, (max_ancho, max_alto))
            return placa_plana
            
    # Si no logra encontrar 4 esquinas limpias, devuelve la placa original
    return img_placa

def segmentar_caracteres(img_placa):
    # 1. Obtener las dimensiones exactas del recorte de la placa
    alto_placa, ancho_placa = img_placa.shape[:2]
    
    gris = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    
    # Binarización con OTSU
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cajas_letras = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        
        # --- FILTROS ESPACIALES ---
        
        # A. Regla para Letras y Números (Verticales y altos)
        es_vertical = h > (w * 1.1) 
        es_suficientemente_alto = h > (alto_placa * 0.45)
        es_caracter_estandar = es_vertical and es_suficientemente_alto
        
        # B. Regla de excepción para el Guion (-)
        # Relajamos la proporción: solo pedimos que sea ligeramente más ancho que alto (o igual)
        es_horizontal = w >= (h * 0.9) 
        
        # Ajustamos el tamaño lógico para atrapar guiones gruesos pero que no sean enormes
        tamano_logico_guion = (h > alto_placa * 0.03) and (w < ancho_placa * 0.15)
        es_guion = es_horizontal and tamano_logico_guion
        
        # ... [Regla C y el resto del código se mantienen igual] ...
        
        # ... [Reglas A y B se mantienen igual] ...
        
        # C. Exclusión mediante el Centroide (Franja Central)
        limite_superior = alto_placa * 0.25
        limite_inferior = alto_placa * 0.75
        
        # Calculamos el punto central vertical del recorte detectado
        centro_y = y + (h / 2)
        
        # Exigimos únicamente que el CENTRO del carácter esté en la franja central
        esta_en_franja_central = (centro_y > limite_superior) and (centro_y < limite_inferior)
        
        # Si es un carácter normal O un guion, Y su centro está en la franja, se procesa
        if (es_caracter_estandar or es_guion) and esta_en_franja_central:
            cajas_letras.append((x, y, w, h))
            
    # Ordenar estrictamente de izquierda a derecha
    cajas_letras = sorted(cajas_letras, key=lambda b: b[0])
    
    recortes_individuales = []
    padding = 3
    
    for (x, y, w, h) in cajas_letras:
        y1, y2 = max(0, y - padding), min(alto_placa, y + h + padding)
        x1, x2 = max(0, x - padding), min(ancho_placa, x + w + padding)
        
        letra_recortada = img_placa[y1:y2, x1:x2]
        
        # --- NUEVO: Deskewing Local (Alineación de la letra) ---
        gris_letra = cv2.cvtColor(letra_recortada, cv2.COLOR_BGR2GRAY)
        _, bin_letra = cv2.threshold(gris_letra, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Encontrar los píxeles blancos que componen la letra
        coordenadas = np.column_stack(np.where(bin_letra > 0))
        if len(coordenadas) > 0:
            # Obtener el ángulo de inclinación exacto de la letra
            rect_min = cv2.minAreaRect(coordenadas)
            angulo_letra = rect_min[-1]
            
            # Ajuste de versión para OpenCV
            if angulo_letra < -45:
                angulo_letra = -(90 + angulo_letra)
            else:
                angulo_letra = -angulo_letra
            
            # Si está inclinada más de 3 grados, la rotamos en su propio eje
            if abs(angulo_letra) > 3:
                (h_l, w_l) = letra_recortada.shape[:2]
                centro = (w_l // 2, h_l // 2)
                M = cv2.getRotationMatrix2D(centro, angulo_letra, 1.0)
                # BORDER_REPLICATE evita que salgan bordes negros al rotar
                letra_recortada = cv2.warpAffine(letra_recortada, M, (w_l, h_l), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # --------------------------------------------------------

        # Redimensión a 128x128 para la CNN (Se mantiene igual a tu código)
        letra_gris = cv2.cvtColor(letra_recortada, cv2.COLOR_BGR2GRAY)
        letra_128 = cv2.resize(letra_gris, (128, 128), interpolation=cv2.INTER_AREA)
        letra_final = cv2.cvtColor(letra_128, cv2.COLOR_GRAY2RGB)
        
        recortes_individuales.append(letra_final)

    return recortes_individuales

def leer_caracter_con_cnn_debug(img_letra):
    # 1. Preprocesamiento
    gris = cv2.cvtColor(img_letra, cv2.COLOR_BGR2GRAY)
    binaria = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1
    )
    kernel = np.ones((2, 2), np.uint8)
    binaria = cv2.dilate(binaria, kernel, iterations=1)

    # Redimensión a 128x128
    img_rgb = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
    img_res = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)

    # Normalización estándar
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_float = (img_res.astype(np.float32) / 255.0 - mean) / std

    # Tensor a GPU
    img_final = np.transpose(img_float, (2, 0, 1))
    tensor_letra = torch.from_numpy(img_final).float().unsqueeze(0).to(device)

    modelo_ocr.eval()
    with torch.no_grad():
        logits = modelo_ocr(tensor_letra)
        probabilidades = F.softmax(logits, dim=1)

        # Obtenemos el Top 3 de predicciones
        top3_probs, top3_indices = torch.topk(probabilidades, 3, dim=1)

        top3_resultados = []
        for prob, idx in zip(top3_probs[0], top3_indices[0]):
            char = ALFABETO[idx.item()]
            char_legal = corregir_normativa_mexicana(char)
            top3_resultados.append(
                (char_legal, idx.item(), prob.item() * 100)
            )

        # Retornamos ganador, imagen procesada y el Top 3 completo
        ganador_char = top3_resultados[0][0]
        ganador_conf = top3_resultados[0][2]
        return ganador_char, ganador_conf, top3_resultados, img_res


def leer_caracter_con_yolo(img_letra):
    """Realiza la inferencia usando el modelo YOLOv8-CLS."""
    res = detector_cls.predict(img_letra, verbose=False)[0]
    # Obtenemos el nombre de la clase con mayor confianza
    return res.names[res.probs.top1]    

def procesar_imagen_cv2(img_cv2):
    """
    Ejecuta la cascada completa: Autos -> Placas -> Segmentación -> OCR
    """
    resultados_finales = []

    # FASE 1: Detectar Autos
    res_autos = detector_autos.predict(img_cv2, conf=0.5, verbose=False)[0]
    
    for caja_auto in res_autos.boxes.xyxy:
        x1, y1, x2, y2 = map(int, caja_auto)
        # recorte_auto = img_cv2[y1:y2, x1:x2]
        # 1. Recorte original de YOLO
        recorte_auto = img_cv2[y1:y2, x1:x2]
        
        # 2. NUEVO: Aplanamos la placa
        recorte_rectificado = rectificar_perspectiva_placa(recorte_auto)
        
        # 3. Se la pasamos a tu función de segmentación
        caracteres = segmentar_caracteres(recorte_rectificado)
        
        # FASE 2: Detectar Placas
        res_placas = detector_placas.predict(recorte_auto, conf=0.4, verbose=False)[0]
        
        for caja_placa in res_placas.boxes.xyxy:
            px1, py1, px2, py2 = map(int, caja_placa)
            recorte_placa = recorte_auto[py1:py2, px1:px2]
            
            # FASE 3: Cortar la placa en caracteres individuales
            caracteres = segmentar_caracteres(recorte_placa)

            texto_placa_cnn = ""
            texto_placa_yolo = ""
            confianzas_caracteres = []
            
            for i, letra_img in enumerate(caracteres):
               # --- LÍNEA DE DEPURACIÓN VISUAL ---
                # Esto guardará cada letra recortada en tu carpeta del proyecto
                cv2.imwrite(f"letra_debug_{i}.jpg", letra_img)
                # ----------------------------------
                
                # Lectura con tu CNN (128x128) - AQUÍ ESTÁ LA CORRECCIÓN
                char_cnn, porcentaje_conf, _, _ = leer_caracter_con_cnn_debug(letra_img)
                
                texto_placa_cnn += char_cnn
                confianzas_caracteres.append(porcentaje_conf)
                
                # Lectura con YOLOv8-CLS
                char_yolo = leer_caracter_con_yolo(letra_img)
                texto_placa_yolo += char_yolo

            if texto_placa_cnn:
                # 4. Calcular el promedio DESPUÉS del for, pero ANTES de guardar el resultado
                if len(confianzas_caracteres) > 0:
                    conf_media = sum(confianzas_caracteres) / len(confianzas_caracteres)
                else:
                    conf_media = 0.0

                resultados_finales.append({
                    'texto_cnn': texto_placa_cnn,
                    'confianza_media': conf_media,
                    'texto_yolo': texto_placa_yolo,
                    'caja': (x1 + px1, y1 + py1, x1 + px2, y1 + py2),
                    'recorte': recorte_placa
                })
    return resultados_finales