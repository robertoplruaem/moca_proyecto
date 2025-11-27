"""
SCRIPT UNIVERSAL DE CONFIGURACIÓN DE PROYECTO (Multi-Dataset)
Permite gestionar la descarga y organización de Autos (Etapa 1) y Placas (Etapa 2).
"""

import os
import shutil
import sys

# --- CONFIGURACIÓN MAESTRA ---
PROJECT_NAME = "Maestria_ALPR"

# Diccionario que define cómo manejar cada dataset
DATASETS_CONFIG = {
    "1": {
        "name": "Autos (Vehicle Detection)",
        "kaggle_id": "seyeon040768/car-detection-dataset",
        "raw_dest": "data/processed/01_autos_original",
        "final_dest": "datasets/01_autos",
        # Este dataset tiene una carpeta contenedora molesta
        "unzip_subfolder": "car_dataset-master", 
        # Mapa de carpetas internas
        "folder_map": {"train": "train", "test": "test", "valid": "val"}
    },
    "2": {
        "name": "Placas (Fares Elmenshawii - Large Dataset)",
        "kaggle_id": "fareselmenshawii/large-license-plate-dataset", # <--- ID Nuevo
        "raw_dest": "data/processed/02_placas_original",
        "final_dest": "datasets/02_placas",
        
        # CAMBIO TÉCNICO IMPORTANTE:
        "unzip_subfolder": None,    # No buscamos una carpeta específica al descomprimir
        "folder_map": {"": ""}      # Truco: Mover TODO el contenido raíz sin filtrar
    }
}

# Estructura de carpetas del proyecto
DIRS = [
    "data/raw",
    "data/processed/01_autos_original",
    "data/processed/02_placas_original",
    "data/processed/03_caracteres_original", # Futuro
    "datasets/01_autos",
    "datasets/02_placas",
    "datasets/03_caracteres",
    "notebooks/01_autos",     # <--- Separación por módulos
    "notebooks/02_placas",
    "notebooks/03_caracteres",
    "src",
    "pipelines",
    "models/vehicle_detector",
    "models/plate_detector",
    "models/ocr_model",
    "production_weights"
]

def create_structure():
    print(f"🏗️  Verificando estructura de directorios...")
    base_path = os.getcwd()
    for folder in DIRS:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        # Crear .gitkeep para persistencia
        gitkeep = os.path.join(path, ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, 'w') as f: pass
    print("   ✅ Estructura base lista.")

def download_dataset(key):
    config = DATASETS_CONFIG[key]
    print(f"\n⬇️  Iniciando proceso para: {config['name']}")
    
    # 1. Verificar Kaggle
    if shutil.which("kaggle") is None:
        print("   ❌ Error: 'kaggle' no instalado. Ejecuta: pip install kaggle")
        return

    # 2. Descargar a data/raw
    download_path = "data/raw"
    print(f"   📡 Descargando {config['kaggle_id']}...")
    try:
        os.system(f"kaggle datasets download -d {config['kaggle_id']} -p {download_path} --unzip")
    except Exception as e:
        print(f"   ❌ Error en descarga: {e}")
        return

    # 3. Mover a Staging (data/processed)
    print(f"   📦 Moviendo a Staging: {config['raw_dest']}...")
    
    # Determinar la raíz de la extracción
    if config['unzip_subfolder']:
        source_root = os.path.join(download_path, config['unzip_subfolder'])
    else:
        source_root = download_path # Si no hay subcarpeta, están en la raíz de raw

    # Verificar si la ruta fuente existe (fallback)
    if not os.path.exists(source_root):
        print(f"   ⚠️ No se encontró la ruta esperada: {source_root}. Usando raíz de raw.")
        source_root = download_path

    # Limpiar destino si ya existe para evitar mezclas
    if os.path.exists(config['raw_dest']):
        shutil.rmtree(config['raw_dest'])
    os.makedirs(config['raw_dest'], exist_ok=True)

    files_moved = False
    
    # Iterar sobre las carpetas que nos interesan
    for k_orig, k_dest in config['folder_map'].items():
        src = os.path.join(source_root, k_orig)
        dst = os.path.join(config['raw_dest'], k_dest)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"      ➡️  {k_orig} -> {dst}")
            files_moved = True
        else:
            print(f"      ⚠️  No se encontró: {k_orig} en {source_root}")

    # 4. Limpieza de data/raw
    # Solo borramos si había subcarpeta contenedora para no borrar otros zips por error
    if config['unzip_subfolder'] and os.path.exists(source_root):
        shutil.rmtree(source_root)
        print("   🧹 Limpieza temporal completada.")

    if files_moved:
        print(f"\n✨ ¡ÉXITO! Datos listos en: {config['raw_dest']}")
        if key == "2":
            print("   ℹ️  NOTA: Este dataset de placas está en XML. Deberás convertirlo a TXT en tu notebook.")
    else:
        print("\n❌ Algo falló. Revisa la carpeta 'data/raw'.")

def menu():
    create_structure()
    while True:
        print("\n--- GESTOR DE DATASETS (MAESTRÍA) ---")
        print("1. Descargar Dataset AUTOS (Etapa 1)")
        print("2. Descargar Dataset PLACAS (Etapa 2)")
        print("3. Salir")
        
        choice = input("Selecciona una opción: ")
        
        if choice in ["1", "2"]:
            download_dataset(choice)
        elif choice == "3":
            print("👋 Hasta luego.")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()