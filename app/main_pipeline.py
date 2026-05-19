# Archivo: app/main_pipeline.py

import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import torch.nn.functional as F # Para ver la confianza

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
ALFABETO = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"

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

def segmentar_caracteres(img_placa):
    gris = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    
    # Binarización con OTSU (Correcto para placas estándar)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cajas_letras = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Ajustamos el filtro: h > w * 1.1 asegura que sea vertical y no ruido cuadrado
        if area > 200 and h > w:
            cajas_letras.append((x, y, w, h))
            
    cajas_letras = sorted(cajas_letras, key=lambda b: b[0])
    
    recortes_individuales = []
    padding = 3
    alto_img, ancho_img = img_placa.shape[:2]

    for (x, y, w, h) in cajas_letras:
        y1, y2 = max(0, y - padding), min(alto_img, y + h + padding)
        x1, x2 = max(0, x - padding), min(ancho_img, x + w + padding)
        
        letra_recortada = img_placa[y1:y2, x1:x2]
        
        # --- NUEVO: REDIMENSIÓN A 64x64 ---
        # 1. Convertimos a gris para la red neuronal
        letra_gris = cv2.cvtColor(letra_recortada, cv2.COLOR_BGR2GRAY)
        
        # 2. Redimensionamos usando INTER_AREA para conservar la nitidez de los bordes
        # Esto es vital para que tu CNN reconozca el patrón entrenado de 200k imágenes
        letra_64 = cv2.resize(letra_gris, (64, 64), interpolation=cv2.INTER_AREA)
        
        # 3. Regresamos a 3 canales si tu red espera RGB (como en el entrenamiento)
        letra_final = cv2.cvtColor(letra_64, cv2.COLOR_GRAY2RGB)
        
        recortes_individuales.append(letra_final)
        
    return recortes_individuales


def leer_caracter_con_cnn(img_letra):
    # 1. Convertir a Gris
    gris = cv2.cvtColor(img_letra, cv2.COLOR_BGR2GRAY)
    
    # CORRECCIÓN: adaptiveThreshold solo devuelve la imagen, no necesita el guion "_"
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 1)
    
    # 2. LIMPIEZA MORFOLÓGICA (Opcional pero recomendado)
    # Esto elimina pequeños puntos negros y suaviza los bordes de la letra
    kernel = np.ones((2,2), np.uint8)
    # binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.dilate(binaria, kernel, iterations=1)

    # 3. CONVERSIÓN A RGB Y REDIMENSIONADO (64x64 como pide tu red)
    img_rgb = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
    img_res = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 4. NORMALIZACIÓN (Estándar de PyTorch/ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_float = (img_res.astype(np.float32) / 255.0 - mean) / std
    
    # 5. Tensor y GPU
    img_final = np.transpose(img_float, (2, 0, 1))
    # tensor_letra = torch.from_numpy(img_final).unsqueeze(0).to(device)
    tensor_letra = torch.from_numpy(img_final).float().unsqueeze(0).to(device)

    modelo_ocr.eval()
    with torch.no_grad():
        logits = modelo_ocr(tensor_letra)
        
        # Aplicamos Softmax para obtener probabilidades entre 0 y 1
        probabilidades = F.softmax(logits, dim=1)
        
        confianza, indice_ganador = torch.max(probabilidades, dim=1)
        
        caracter_original = ALFABETO[indice_ganador.item()]
        caracter_legal = corregir_normativa_mexicana(caracter_original)
        
        # Devolvemos el carácter y la confianza (convertida a porcentaje)
        return caracter_legal, confianza.item() * 100

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
        recorte_auto = img_cv2[y1:y2, x1:x2]
        
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
                
                # Lectura con tu CNN (64x64)
                char_cnn, porcentaje_conf = leer_caracter_con_cnn(letra_img)
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