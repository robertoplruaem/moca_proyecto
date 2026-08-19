import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from customOCR_CNN import CustomOCR_CNN

# --- 1. CONFIGURACIÓN GLOBAL ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Procesador PyTorch configurado en: {device}")

ALFABETO = "-0123456789ABCDEFGHJKLMNPRSTUVWXYZ"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_AUTOS = os.path.join(BASE_DIR, '../production_weights/01_autos_best.pt')
RUTA_PLACAS = os.path.join(BASE_DIR, '../production_weights/02_placas_best.pt')
RUTA_OCR = os.path.join(BASE_DIR, '../production_weights/03_caracteres_best.pth')
RUTA_YOLO_CLS = os.path.join(BASE_DIR, '../production_weights/03_yolo_cls_best.pt')

# --- 2. INICIALIZAR MODELOS ---
print("Cargando modelos en memoria...")
try:
    detector_autos = YOLO(RUTA_AUTOS)
    detector_placas = YOLO(RUTA_PLACAS)
    detector_cls = YOLO(RUTA_YOLO_CLS)
    print("Modelos YOLO listos.")
except Exception as e:
    print(f"Error cargando YOLO: {e}")

try:
    modelo_ocr = CustomOCR_CNN(num_classes=len(ALFABETO)) 
    pesos = torch.load(RUTA_OCR, map_location=device, weights_only=True)
    modelo_ocr.load_state_dict(pesos)
    modelo_ocr.to(device)
    modelo_ocr.eval()
    print("Red Neuronal CNN OCR lista y cargada en GPU.")
except Exception as e:
    print(f"Error crítico cargando la CNN: {e}")

# --- 3. FUNCIONES AUXILIARES ---
def corregir_normativa_mexicana(caracter):
    mapeo = {'I': '1', 'Ñ': 'N', 'O': '0', 'Q': '0'}
    return mapeo.get(caracter, caracter)

def rectificar_perspectiva_placa(img_placa):
    gris = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(blur, 50, 150)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contornos:
        c_max = max(contornos, key=cv2.contourArea)
        perimetro = cv2.arcLength(c_max, True)
        aproximacion = cv2.approxPolyDP(c_max, 0.05 * perimetro, True)
        
        if len(aproximacion) == 4:
            pts = aproximacion.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            max_ancho = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            max_alto = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            
            dst = np.array([[0, 0], [max_ancho - 1, 0], [max_ancho - 1, max_alto - 1], [0, max_alto - 1]], dtype="float32")
            matriz = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(img_placa, matriz, (max_ancho, max_alto))
            
    return img_placa

def corregir_polaridad_placa(img_placa):
    """
    Analiza la luminosidad de la franja central de la placa.
    Si detecta un fondo oscuro (placa de Guerrero/Edomex), 
    invierte los colores (bitwise_not) para simular una placa blanca estándar.
    """
    alto, ancho = img_placa.shape[:2]
    
    # Extraemos exclusivamente la franja central (evitando los bordes blancos de la placa)
    # Tomamos del 30% al 70% del alto, donde con seguridad están las letras y el fondo dominante
    franja_central = img_placa[int(alto * 0.30):int(alto * 0.70), :]
    
    # Convertimos a escala de grises para medir la intensidad de luz
    gris_central = cv2.cvtColor(franja_central, cv2.COLOR_BGR2GRAY)
    
    # Calculamos la media de luminosidad (0 = Negro absoluto, 255 = Blanco absoluto)
    luminosidad_promedio = np.mean(gris_central)
    
    # Un fondo guinda/oscuro suele tener una media inferior a 110.
    # Un fondo blanco/claro suele tener una media superior a 150.
    if luminosidad_promedio < 110:
        # bitwise_not invierte la matriz: lo negro se hace blanco y viceversa.
        # El guinda se hace claro y las letras blancas se hacen negras.
        placa_invertida = cv2.bitwise_not(img_placa)
        return placa_invertida
        
    # Si la placa es clara, la devolvemos intacta
    return img_placa

def preprocesar_y_estandarizar_caracter(recorte_bgr, tamano_destino=128):
    """Binariza, limpia fracturas y centra el carácter en un lienzo blanco estandarizado."""
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    #binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 1. (Opcional pero muy recomendado) Aplicar un desenfoque mediano ligero antes 
    # para "matar" los píxeles de ruido aislados sin difuminar los bordes duros de la letra.
    gris = cv2.medianBlur(gris, 3)

    # 2. Binarización con radio ampliado (21) y mayor exigencia de contraste (5)
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morfologica = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    alto, ancho = morfologica.shape
    factor = tamano_destino / max(alto, ancho)
    nuevo_alto, nuevo_ancho = int(alto * factor), int(ancho * factor)
    redimensionada = cv2.resize(morfologica, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    
    lienzo = np.ones((tamano_destino, tamano_destino), dtype=np.uint8) * 255
    y_offset = (tamano_destino - nuevo_alto) // 2
    x_offset = (tamano_destino - nuevo_ancho) // 2
    lienzo[y_offset:y_offset+nuevo_alto, x_offset:x_offset+nuevo_ancho] = redimensionada
    
    return cv2.cvtColor(lienzo, cv2.COLOR_GRAY2RGB)

def segmentar_caracteres_crudos(img_placa):
    """Encuentra y extrae los recortes de los caracteres sin procesarlos matemáticamente."""
    alto_placa, ancho_placa = img_placa.shape[:2]
    gris = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cajas_letras = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        es_caracter_estandar = (h > w * 1.1) and (h > alto_placa * 0.45)
        es_guion = (w >= h * 0.9) and (h > alto_placa * 0.03) and (w < ancho_placa * 0.15)
        
        centro_y = y + (h / 2)
        esta_en_franja_central = (centro_y > alto_placa * 0.25) and (centro_y < alto_placa * 0.75)
        
        if (es_caracter_estandar or es_guion) and esta_en_franja_central:
            cajas_letras.append((x, y, w, h))
            
    cajas_letras = sorted(cajas_letras, key=lambda b: b[0])
    recortes_individuales = []
    padding = 3
    
    for (x, y, w, h) in cajas_letras:
        y1, y2 = max(0, y - padding), min(alto_placa, y + h + padding)
        x1, x2 = max(0, x - padding), min(ancho_placa, x + w + padding)
        letra_recortada = img_placa[y1:y2, x1:x2]
        
        gris_letra = cv2.cvtColor(letra_recortada, cv2.COLOR_BGR2GRAY)
        _, bin_letra = cv2.threshold(gris_letra, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        coordenadas = np.column_stack(np.where(bin_letra > 0))
        
        if len(coordenadas) > 0:
            rect_min = cv2.minAreaRect(coordenadas)
            angulo_letra = rect_min[-1]
            angulo_letra = -(90 + angulo_letra) if angulo_letra < -45 else -angulo_letra
            
            if abs(angulo_letra) > 3:
                h_l, w_l = letra_recortada.shape[:2]
                M = cv2.getRotationMatrix2D((w_l // 2, h_l // 2), angulo_letra, 1.0)
                letra_recortada = cv2.warpAffine(letra_recortada, M, (w_l, h_l), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        recortes_individuales.append(letra_recortada)

    return recortes_individuales

def inferencia_cnn(img_rgb_128):
    """Realiza la conversión a tensor y la predicción en PyTorch."""
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_float = (img_rgb_128.astype(np.float32) / 255.0 - mean) / std
    img_final = np.transpose(img_float, (2, 0, 1))
    tensor_letra = torch.from_numpy(img_final).float().unsqueeze(0).to(device)

    with torch.no_grad():
        probabilidades = F.softmax(modelo_ocr(tensor_letra), dim=1)
        top_prob, top_idx = torch.topk(probabilidades, 1, dim=1)
        ganador_char = corregir_normativa_mexicana(ALFABETO[top_idx.item()])
        return ganador_char, top_prob.item() * 100

def inferencia_yolo(img_rgb_128):
    res = detector_cls.predict(img_rgb_128, verbose=False)[0]
    return res.names[res.probs.top1]

# --- 4. PIPELINE PRINCIPAL ---
def procesar_imagen_cv2(img_cv2):
    resultados_finales = []
    res_autos = detector_autos.predict(img_cv2, conf=0.5, verbose=False)[0]
    
    for caja_auto in res_autos.boxes.xyxy:
        x1, y1, x2, y2 = map(int, caja_auto)
        recorte_auto = img_cv2[y1:y2, x1:x2]
        
        res_placas = detector_placas.predict(recorte_auto, conf=0.40, verbose=False)[0]
        for caja_placa in res_placas.boxes.xyxy:
            px1, py1, px2, py2 = map(int, caja_placa)
            recorte_placa = recorte_auto[py1:py2, px1:px2]
            
            # 1. Aplanamos la geometría
            placa_rectificada = rectificar_perspectiva_placa(recorte_placa)
            
            # 2. Intento A: Segmentación asumiendo placa blanca estándar
            caracteres_crudos = segmentar_caracteres_crudos(placa_rectificada)
            
            # 3. NUEVO: Fallback de Polaridad Dinámica
            # Una placa mexicana tiene entre 5 y 7 caracteres. Si encuentra 3 o menos, 
            # es muy probable que la binarización haya fallado por fondo oscuro o sombra.
            if len(caracteres_crudos) < 4:
                # Invertimos los colores matemáticamente
                placa_invertida = cv2.bitwise_not(placa_rectificada)
                
                # Intento B: Segmentación asumiendo placa oscura
                caracteres_alternativos = segmentar_caracteres_crudos(placa_invertida)
                
                # Si la versión invertida logra "ver" más letras, nos quedamos con esa
                if len(caracteres_alternativos) > len(caracteres_crudos):
                    caracteres_crudos = caracteres_alternativos

            texto_cnn, texto_yolo = "", ""
            confianzas_caracteres = []
            detalles_caracteres = []
            
            for i, recorte_crudo in enumerate(caracteres_crudos):
                # -----------------------------------------------------------
                # 1. CORRIENTE CNN: Preprocesamiento matemático (Binarizado)
                # -----------------------------------------------------------
                img_lista_128_cnn = preprocesar_y_estandarizar_caracter(recorte_crudo, tamano_destino=128)
                
                char_cnn, conf = inferencia_cnn(img_lista_128_cnn)
                texto_cnn += char_cnn
                confianzas_caracteres.append(conf)
                
                # -----------------------------------------------------------
                # 2. CORRIENTE YOLO: Preprocesamiento fotográfico (Texturas)
                # -----------------------------------------------------------
                # Replicamos el trato exacto que tenía tu código original:
                gris_yolo = cv2.cvtColor(recorte_crudo, cv2.COLOR_BGR2GRAY)
                img_128_yolo = cv2.resize(gris_yolo, (128, 128), interpolation=cv2.INTER_AREA)
                img_yolo_final = cv2.cvtColor(img_128_yolo, cv2.COLOR_GRAY2RGB)
                
                texto_yolo += inferencia_yolo(img_yolo_final)
                
                # -----------------------------------------------------------
                # Guardar evidencia visual (Mostramos la versión de la CNN en la web)
                # -----------------------------------------------------------
                detalles_caracteres.append({
                    'imagen_128': img_lista_128_cnn,
                    'char_cnn': char_cnn,
                    'conf': conf
                })

            conf_media = sum(confianzas_caracteres) / len(confianzas_caracteres) if confianzas_caracteres else 0.0

            if texto_cnn:
                resultados_finales.append({
                    'texto_cnn': texto_cnn,
                    'confianza_media': conf_media,
                    'texto_yolo': texto_yolo,
                    'caja': (x1 + px1, y1 + py1, x1 + px2, y1 + py2),
                    'recorte': recorte_placa,
                    'recorte_auto': recorte_auto,
                    'detalles': detalles_caracteres
                })
    return resultados_finales