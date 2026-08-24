import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
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

# --- REGISTRO DE TIEMPOS GLOBAL ---
registro_tiempos = {
    'det_vehiculo': [],
    'det_placa': [],
    'preprocesamiento_cv2': [],
    'ocr_dual': [],
    'total_pipeline': []
}
vehiculos_procesados = 0

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
    alto, ancho = img_placa.shape[:2]
    franja_central = img_placa[int(alto * 0.30):int(alto * 0.70), :]
    gris_central = cv2.cvtColor(franja_central, cv2.COLOR_BGR2GRAY)
    luminosidad_promedio = np.mean(gris_central)
    
    if luminosidad_promedio < 110:
        placa_invertida = cv2.bitwise_not(img_placa)
        return placa_invertida
        
    return img_placa

def preprocesar_y_estandarizar_caracter(recorte_bgr, tamano_destino=128):
    gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 3)
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

# --- 4. PIPELINE PRINCIPAL (Con medición de latencia) ---
def procesar_imagen_cv2(img_cv2):
    global vehiculos_procesados
    resultados_finales = []
    
    t_inicio_total = time.perf_counter()
    
    t0 = time.perf_counter()
    res_autos = detector_autos.predict(img_cv2, conf=0.5, verbose=False)[0]
    registro_tiempos['det_vehiculo'].append((time.perf_counter() - t0) * 1000)
    
    for caja_auto in res_autos.boxes.xyxy:
        x1, y1, x2, y2 = map(int, caja_auto)
        recorte_auto = img_cv2[y1:y2, x1:x2]
        
        t0 = time.perf_counter()
        res_placas = detector_placas.predict(recorte_auto, conf=0.40, verbose=False)[0]
        registro_tiempos['det_placa'].append((time.perf_counter() - t0) * 1000)
        
        for caja_placa in res_placas.boxes.xyxy:
            px1, py1, px2, py2 = map(int, caja_placa)
            recorte_placa = recorte_auto[py1:py2, px1:px2]
            
            t0 = time.perf_counter()
            placa_rectificada = rectificar_perspectiva_placa(recorte_placa)
            caracteres_crudos = segmentar_caracteres_crudos(placa_rectificada)
            
            # 3. NUEVO: Fallback de Polaridad Dinámica
            if len(caracteres_crudos) < 4:

                # --- IMPRIMIR CONTEO INICIAL ---
                print("\n[ALERTA] Posible placa oscura detectada.")
                print(f"-> Intento A (Estándar): Solo se aislaron {len(caracteres_crudos)} caracteres.")
                
                # --- INICIO DE EXTRACCIÓN DE IMÁGENES PARA LA TESIS ---
                # 1. Guardar la placa original (fondo oscuro) rectificada
                cv2.imwrite("tesis_01_placa_oscura_original.jpg", placa_rectificada)
                
                # 2. Guardar la binarización errónea inicial (simulando lo que vio segmentar_caracteres_crudos)
                gris_erroneo = cv2.cvtColor(placa_rectificada, cv2.COLOR_BGR2GRAY)
                _, binaria_erronea = cv2.threshold(gris_erroneo, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cv2.imwrite("tesis_02_binarizacion_erronea.jpg", binaria_erronea)
                
                # Invertimos los colores matemáticamente
                placa_invertida = cv2.bitwise_not(placa_rectificada)
                
                # 3. Guardar la placa con colores invertidos
                cv2.imwrite("tesis_03_placa_invertida.jpg", placa_invertida)
                
                # 4. Guardar la binarización exitosa
                gris_exitoso = cv2.cvtColor(placa_invertida, cv2.COLOR_BGR2GRAY)
                _, binaria_exitosa = cv2.threshold(gris_exitoso, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cv2.imwrite("tesis_04_binarizacion_exitosa.jpg", binaria_exitosa)
                # --- FIN DE EXTRACCIÓN DE IMÁGENES ---                

                # Intento B: Segmentación asumiendo placa oscura
                caracteres_alternativos = segmentar_caracteres_crudos(placa_invertida)

                # --- IMPRIMIR CONTEO SECUNDARIO ---
                print(f"-> Intento B (Invertido): Se aislaron {len(caracteres_alternativos)} caracteres.")
                
                # Si la versión invertida logra "ver" más letras, nos quedamos con esa
                if len(caracteres_alternativos) > len(caracteres_crudos):
                    caracteres_crudos = caracteres_alternativos
            
            
            registro_tiempos['preprocesamiento_cv2'].append((time.perf_counter() - t0) * 1000)

            texto_cnn, texto_yolo = "", ""
            confianzas_caracteres = []
            detalles_caracteres = []
            
            t0 = time.perf_counter()
            for i, recorte_crudo in enumerate(caracteres_crudos):
                img_lista_128_cnn = preprocesar_y_estandarizar_caracter(recorte_crudo, tamano_destino=128)
                char_cnn, conf = inferencia_cnn(img_lista_128_cnn)
                texto_cnn += char_cnn
                confianzas_caracteres.append(conf)
                
                # gris_yolo = cv2.cvtColor(recorte_crudo, cv2.COLOR_BGR2GRAY)
                # img_128_yolo = cv2.resize(gris_yolo, (128, 128), interpolation=cv2.INTER_AREA)
                # img_yolo_final = cv2.cvtColor(img_128_yolo, cv2.COLOR_GRAY2RGB)
                # texto_yolo += inferencia_yolo(img_yolo_final)
                
                detalles_caracteres.append({
                    'imagen_128': img_lista_128_cnn,
                    'char_cnn': char_cnn,
                    'conf': conf
                })
            # registro_tiempos['ocr_dual'].append((time.perf_counter() - t0) * 1000)
            registro_tiempos['ocr_dual'].append((time.perf_counter() - t0) * 1000)

            conf_media = sum(confianzas_caracteres) / len(confianzas_caracteres) if confianzas_caracteres else 0.0

            if texto_cnn:
                resultados_finales.append({
                    'texto_cnn': texto_cnn,
                    'confianza_media': conf_media,
                    #'texto_yolo': texto_yolo,
                    'caja': (x1 + px1, y1 + py1, x1 + px2, y1 + py2),
                    'recorte': recorte_placa,
                    'recorte_auto': recorte_auto,
                    'detalles': detalles_caracteres
                })
                vehiculos_procesados += 1
                registro_tiempos['total_pipeline'].append((time.perf_counter() - t_inicio_total) * 1000)
                
    return resultados_finales

# --- 5. FUNCIÓN DE PRUEBA Y REPORTE DE LATENCIA ---
def ejecutar_prueba_rendimiento(carpeta_imagenes):
    archivos = [f for f in os.listdir(carpeta_imagenes) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not archivos:
        print("No se encontraron imágenes en el directorio proporcionado.")
        return

    print("\nIniciando Warm-up (Precalentamiento de GPU)...")
    img_warmup = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(2):
        procesar_imagen_cv2(img_warmup)
    
    # Limpiamos los registros contaminados por el Warm-up
    for key in registro_tiempos:
        registro_tiempos[key].clear()
    global vehiculos_procesados
    vehiculos_procesados = 0
    
    print(f"Iniciando inferencia sobre {len(archivos)} imágenes...")
    
    for idx, archivo in enumerate(archivos):
        ruta_img = os.path.join(carpeta_imagenes, archivo)
        img_cv2 = cv2.imread(ruta_img)
        if img_cv2 is not None:
            procesar_imagen_cv2(img_cv2)
            print(f"Procesada {idx+1}/{len(archivos)}: {archivo}", end='\r')
            
    print("\n\n" + "="*60)
    print("     REPORTE DE TIEMPOS DE INFERENCIA (MILISEGUNDOS)     ")
    print("="*60)
    print(f"Total de vehículos exitosamente procesados: {vehiculos_procesados}")

    if vehiculos_procesados > 0:
        prom_auto = np.mean(registro_tiempos['det_vehiculo'])
        prom_placa = np.mean(registro_tiempos['det_placa'])
        prom_cv2 = np.mean(registro_tiempos['preprocesamiento_cv2'])
        prom_ocr = np.mean(registro_tiempos['ocr_dual'])
        prom_total = np.mean(registro_tiempos['total_pipeline'])
        fps_promedio = 1000 / prom_total
        
        print(f"1. Detección de Vehículo (YOLOv11):          {prom_auto:.2f} ms")
        print(f"2. Detección de Placa en Recorte (YOLO11n):  {prom_placa:.2f} ms")
        print(f"3. Rectificación y Preprocesamiento (CV2):   {prom_cv2:.2f} ms")
        print(f"4. Inferencia OCR Dual (CNN + YOLO-cls):     {prom_ocr:.2f} ms")
        print("-" * 60)
        print(f"TIEMPO TOTAL POR VEHÍCULO (Latencia E2E):    {prom_total:.2f} ms")
        print(f"RENDIMIENTO ESTIMADO EN TIEMPO REAL:         {fps_promedio:.2f} FPS")
    print("="*60)

# Para ejecutar la prueba, puedes llamar a esta función al final del archivo o desde otra terminal:
ejecutar_prueba_rendimiento(os.path.join(BASE_DIR, '../datasets/05_placa_oscura'))