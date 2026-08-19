import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import mysql.connector
from datetime import datetime
import time

# --- IMPORTACIONES ACTUALIZADAS DEL PIPELINE ---
from main_pipeline import (
    procesar_imagen_cv2, 
    segmentar_caracteres_crudos, 
    preprocesar_y_estandarizar_caracter, 
    inferencia_cnn
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Reconocimiento de placas - MOCA UAEM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FUNCIONES DE PERSISTENCIA (BASE DE DATOS) ---
def guardar_deteccion(texto_cnn, confianza_cnn, texto_yolo, tiempo_ms, imagen_cv2):
    """
    Registra los datos en MySQL y guarda el recorte físico de la placa.
    """
    try:
        conexion = mysql.connector.connect(
            host="localhost",
            user="admin_moca",      
            password="Temporal",    
            database="proyecto_moca" 
        )
        cursor = conexion.cursor()

        ruta_carpeta = os.path.join("static", "capturas")
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)
        
        nombre_archivo = f"placa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
        cv2.imwrite(ruta_completa, imagen_cv2)

        sql = """INSERT INTO registros_placas 
                 (placa_texto_cnn, confianza_cnn, placa_texto_yolo, ruta_imagen, tiempo_procesamiento) 
                 VALUES (%s, %s, %s, %s, %s)"""
        
        ruta_relativa = f"static/capturas/{nombre_archivo}"
        valores = (texto_cnn, float(confianza_cnn), texto_yolo, ruta_relativa, float(tiempo_ms))
        
        cursor.execute(sql, valores)
        conexion.commit()
        
        cursor.close()
        conexion.close()
        return True
    except Exception as e:
        st.error(f"Error de conexión a base de datos: {e}")
        return False

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #0070b8; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .titulo-inst { color: #252e61; font-weight: bold; margin-bottom: 0px; }
    .sub-inst { color: #0070b8; margin-top: 0px; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    try:
        st.image(Image.open("logo_uaem.png"), width=120)
        st.image(Image.open("logo_fcaei.png"), width=120)
    except:
        pass # Falla silenciosa si no hay logos

with col_titulo:
    st.markdown("<h1 class='titulo-inst'>Universidad Autónoma del Estado de Morelos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-inst'>Facultad de Contaduría, Administración e Informática</h3>", unsafe_allow_html=True)
    st.markdown("#### Proyecto MOCA: Reconocimiento de Placas Vehiculares")

st.divider()

# --- INTERFAZ PRINCIPAL (VISTA DIVIDIDA) ---
col_controles, col_visor = st.columns([1, 2], gap="large")

with col_controles:
    st.markdown("### Panel de Control")
    archivo_subido = st.file_uploader("Cargar imagen vehicular", type=['jpg', 'jpeg', 'png'])
    boton_analizar = st.button("Ejecutar Análisis", use_container_width=True, type="primary")

with col_visor:
    st.markdown("### Visor Original")
    if archivo_subido is not None:
        imagen_pil = Image.open(archivo_subido).convert('RGB')
        imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        st.image(imagen_pil, use_container_width=True)
    else:
        st.info("Esperando imagen para visualización.")

st.divider()

# --- EJECUCIÓN DEL PIPELINE ---
if archivo_subido is not None and boton_analizar:
    st.markdown("### Resultados de Inferencia")
    
    with st.spinner("Procesando redes neuronales..."):
        inicio_ms = time.time()
        resultados = procesar_imagen_cv2(imagen_cv2)
        tiempo_total = (time.time() - inicio_ms) * 1000.0
        
    if not resultados:
        st.error("No se detectaron placas vehiculares en la imagen proporcionada.")
    else:
        tiempo_por_placa = tiempo_total / len(resultados)
        
        for idx, res in enumerate(resultados):
            st.markdown(f"#### Detección Vehicular #{idx + 1}")
            
            # --- GALERÍA DE EVIDENCIAS: AUTO | PLACA | MÉTRICAS ---
            col_auto, col_placa, col_metricas = st.columns([1.5, 1.5, 2])
            
            with col_auto:
                st.markdown("**Vehículo**")
                auto_rgb = cv2.cvtColor(res['recorte_auto'], cv2.COLOR_BGR2RGB)
                st.image(auto_rgb, use_container_width=True)
                
            with col_placa:
                st.markdown("**Placa Rectificada**")
                placa_rgb = cv2.cvtColor(res['recorte'], cv2.COLOR_BGR2RGB)
                st.image(placa_rgb, use_container_width=True)
            
            with col_metricas:
                st.markdown("**Lectura**")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="CNN Propuesta", value=res['texto_cnn'])
                with m2:
                    st.metric(label="YOLOv11-cls", value=res['texto_yolo'])
                
                # Un solo bloque de registro y alertas
                if guardar_deteccion(res['texto_cnn'], res['confianza_media'], res['texto_yolo'], tiempo_por_placa, res['recorte']):
                    st.success(f"Registrado en DB. Tiempo: {tiempo_por_placa:.1f} ms.")                    
                if res['confianza_media'] < 80:
                    st.warning("Confianza de lectura CNN inferior al 80%.")

           # --- DEPURACIÓN OCR SIN RE-PROCESAMIENTO ---
            with st.expander("Inspección Geométrica de Caracteres (CNN)"):
                detalles = res.get('detalles', [])
                
                if detalles:
                    cols = st.columns(len(detalles))
                    for i, detalle in enumerate(detalles):
                        with cols[i]:
                            # Mostramos la imagen exacta que usó el pipeline
                            img_rgb = cv2.cvtColor(detalle['imagen_128'], cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                            
                            # Mostramos la predicción exacta de esa imagen
                            st.markdown(f"**`{detalle['char_cnn']}`**")
                            st.caption(f"{detalle['conf']:.1f}%")
                else:
                    st.info("La segmentación no logró aislar caracteres válidos.")
                    
            st.write("---")

# --- PIE DE PÁGINA ---
st.caption("Elaborado por: Roberto Pablo López Romero | Facultad de Contaduría, Administración e Informática - UAEM | 2026")