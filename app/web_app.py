import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Importamos el pipeline que ya configuramos con la doble inferencia
from main_pipeline import procesar_imagen_cv2

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema de Reconocimiento de Placas - México",
    page_icon="🇲🇽",
    layout="wide"
)

# --- ESTILOS PERSONALIZADOS (Colores institucionales) ---
# Usamos la paleta actualizada: Evalúa (#9dbcc9), Data (#0070b8), Planea (#252e61)
st.markdown(f"""
    <style>
    .main {{ background-color: #f5f7f9; }}
    .stMetric {{ 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #0070b8; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    .titulos {{ color: #252e61; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

st.title("🇲🇽 Identificación de Placas Vehiculares")
st.markdown("### Comparativa de Modelos: Custom CNN vs YOLOv8-CLS")

# --- BARRA LATERAL: CONFIGURACIÓN ---
with st.sidebar:
    st.header("Configuración")
    confianza = st.slider("Confianza de Detección (YOLO)", 0.0, 1.0, 0.45)
    st.info("Este sistema aplica la normativa SCT: excluye I, Ñ, O, Q.")
    
    st.subheader("Hardware Status")
    # Verificación rápida de la GPU para el usuario
    import torch
    if torch.cuda.is_available():
        st.success(f"🚀 GPU Activa: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ Ejecutando en CPU")

# --- CARGA DE ARCHIVOS ---
archivo_subido = st.file_uploader("Sube una imagen de un vehículo o placa", type=['jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    # Convertir archivo a formato OpenCV
    imagen_pil = Image.open(archivo_subido).convert('RGB')
    imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    
    st.image(imagen_pil, caption="Imagen Original", use_container_width=True)
    
    if st.button("🚀 Iniciar Procesamiento"):
        with st.spinner("Procesando con RTX 4070 Ti..."):
            # Llamada al pipeline que devuelve texto_cnn y texto_yolo
            resultados = procesar_imagen_cv2(imagen_cv2)
            
        if not resultados:
            st.error("No se detectaron placas en la imagen.")
        else:
            st.success(f"Se detectaron {len(resultados)} placa(s).")
            
            for idx, res in enumerate(resultados):
                st.divider()
                st.subheader(f"Placa #{idx + 1}")
                
                col_img, col_metrics = st.columns([1, 2])
                
                with col_img:
                    # Mostrar el recorte de la placa detectada
                    recorte_rgb = cv2.cvtColor(res['recorte'], cv2.COLOR_BGR2RGB)
                    st.image(recorte_rgb, caption="Recorte de Placa", use_container_width=True)
                
                with col_metrics:
                    m1, m2 = st.columns(2)
                    
                    # --- RESULTADO TU CUSTOM CNN ---
                    with m1:
                        st.metric(
                            label="Tu Custom CNN (Precisión: 98.88%)", 
                            value=res['texto_cnn'],
                            delta=f"{res['confianza_media']:.2f}% Confianza"
                        )
                        st.caption("⚡ Latencia: 0.0091 ms/char")
                        st.write("✅ Normativa SCT Aplicada")
                    
                    # --- RESULTADO YOLOv8-CLS ---
                    with m2:
                        st.metric(
                            label="YOLOv8-CLS (Precisión: 96.28%)", 
                            value=res['texto_yolo']
                        )
                        st.caption("🐢 Latencia: 0.1308 ms/char")
                        st.write("❌ Sin filtros de normativa")

                # Mostrar la ubicación en la imagen original
                st.info(f"Coordenadas de la caja: {res['caja']}")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Desarrollado para el proyecto de Administración Informática - Optimización de Modelos de Deep Learning.")