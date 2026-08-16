import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import mysql.connector
from datetime import datetime
import time
# Cambia tu línea de importación actual por esta:
from main_pipeline import procesar_imagen_cv2, segmentar_caracteres, leer_caracter_con_cnn_debug

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Reconocimiento de placas - MOCA UAEM",
    # page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FUNCIONES DE PERSISTENCIA (BASE DE DATOS) ---
def guardar_deteccion(texto_cnn, confianza_cnn, texto_yolo, tiempo_ms, imagen_cv2):
    """
    Registra los datos en MySQL y guarda el recorte físico.
    """
    try:
        # Asegúrate de que el usuario y la base de datos coincidan con tu configuración de MySQL
        conexion = mysql.connector.connect(
            host="localhost",
            user="admin_moca",      
            password="Temporal",    
            database="proyecto_moca" 
        )
        cursor = conexion.cursor()

        # Guardar la imagen físicamente en la carpeta del servidor
        ruta_carpeta = os.path.join("static", "capturas")
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)
        
        nombre_archivo = f"placa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
        cv2.imwrite(ruta_completa, imagen_cv2)

        # --- NUEVA CONSULTA SQL ---
        sql = """INSERT INTO registros_placas 
                 (placa_texto_cnn, confianza_cnn, placa_texto_yolo, ruta_imagen, tiempo_procesamiento) 
                 VALUES (%s, %s, %s, %s, %s)"""
        
        ruta_relativa = f"static/capturas/{nombre_archivo}"
        
        # Pasamos los 5 valores a la tupla
        valores = (texto_cnn, float(confianza_cnn), texto_yolo, ruta_relativa, float(tiempo_ms))
        
        cursor.execute(sql, valores)
        conexion.commit()
        
        cursor.close()
        conexion.close()
        return True
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return False

# --- ESTILOS PERSONALIZADOS ---
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ display: none; }}
    .main {{ background-color: #f5f7f9; }}
    .stMetric {{ 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #0070b8; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    .titulo-inst {{ color: #252e61; font-weight: bold; margin-bottom: 0px; }}
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    try:
        st.image(Image.open("logo_uaem.png"), width=120)
        st.image(Image.open("logo_fcaei.png"), width=120)
    except:
        st.info("Logos no encontrados.")

with col_titulo:
    st.markdown("<h1 class='titulo-inst'>Universidad Autónoma del Estado de Morelos</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#0070b8;'>Facultad de Contaduría, Administración e Informática</h2>", unsafe_allow_html=True)
    st.subheader("REconocimiento de placas vehiculares")

st.divider()

# --- CARGA Y PROCESAMIENTO ---
archivo_subido = st.file_uploader("Seleccione una imagen para procesar", type=['jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    
    st.image(imagen_pil, caption="Imagen original", width=600)

    if st.button("Analizar"):
        with st.spinner("Analizando..."):
            # Medir el tiempo de inicio
            inicio_ms = time.time()
            
            # Resultados del pipeline (34 clases)
            resultados = procesar_imagen_cv2(imagen_cv2)
            
            # Calcular tiempo total en milisegundos
            tiempo_total = (time.time() - inicio_ms) * 1000.0
            
        if not resultados:
            st.error("No se detectaron placas en la imagen.")
        else:
            # Dividimos el tiempo entre la cantidad de placas detectadas en la foto
            tiempo_por_placa = tiempo_total / len(resultados)
            
            for idx, res in enumerate(resultados):
                st.markdown(f"### Placa Detectada #{idx + 1}")
                
                col_img, col_metrics = st.columns([1, 2])
                
                with col_img:
                    recorte_rgb = cv2.cvtColor(res['recorte'], cv2.COLOR_BGR2RGB)
                    st.image(recorte_rgb, caption="Segmentación", use_container_width=True)
                
                with col_metrics:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="Red propuesta", value=res['texto_cnn'])
                    with m2:
                        st.metric(label="YOLOv11-cls", value=res['texto_yolo'])
                    
                    # --- NUEVA LLAMADA A LA FUNCIÓN CON LOS PARÁMETROS COMPLETOS ---
                    if guardar_deteccion(res['texto_cnn'], res['confianza_media'], res['texto_yolo'], tiempo_por_placa, res['recorte']):
                        st.success(f"Placa {res['texto_cnn']} registrada exitosamente en {tiempo_por_placa:.2f} ms.")                    
                    if res['confianza_media'] < 80:
                        st.warning("Confianza baja. Verifique manualmente.")

                    # ... [Código existente en web_app.py] ...
                    
                    # REGISTRO AUTOMÁTICO AL TERMINAR EL PROCESAMIENTO
                    if guardar_deteccion(res['texto_cnn'], res['confianza_media'], res['texto_yolo'], tiempo_por_placa, res['recorte']):
                        st.success(f"Placa {res['texto_cnn']} registrada exitosamente en {tiempo_por_placa:.2f} ms.")                    
                    if res['confianza_media'] < 80:
                        st.warning("Confianza baja. Verifique manualmente.")

                    # =========================================================
                    # --- AQUÍ COMIENZA EL CÓDIGO DEL PASO 2 (DEPURACIÓN) ---
                    # =========================================================
                    with st.expander("🔍 Inspección Carácter por Carácter (Depuración OCR)"):
                        # Extraemos los caracteres individuales usando el recorte de esta placa
                        caracteres = segmentar_caracteres(res['recorte'])
                        
                        # Creamos las columnas dinámicamente según la cantidad de letras
                        if len(caracteres) > 0:
                            cols = st.columns(len(caracteres))
                            for i, letra_img in enumerate(caracteres):
                                char, conf, top3, img_proc = leer_caracter_con_cnn_debug(letra_img)
                                with cols[i]:
                                    # Muestra lo que realmente recibe la CNN
                                    st.image(img_proc, caption=f"Pos #{i+1}", width=60)
                                    st.markdown(f"**Pred:** `{char}` ({conf:.1f}%)")

                                    # Muestra Top 3 alternativas
                                    st.caption(
                                        f"1: {top3[0][0]} ({top3[0][2]:.1f}%)\n\n"
                                        f"2: {top3[1][0]} ({top3[1][2]:.1f}%)\n\n"
                                        f"3: {top3[2][0]} ({top3[2][2]:.1f}%)"
                                    )
                        else:
                            st.info("No se pudieron aislar caracteres en este recorte.")
                    # =========================================================
                

    
st.divider()
st.caption("Elaborado por: Roberto Pablo López Romero | Facultad de Contaduría, Administración e Informática - UAEM | 2026")