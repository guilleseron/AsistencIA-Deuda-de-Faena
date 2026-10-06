# Utilización de imagen oficial de Python ligera
FROM python:3.11-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Actualización del sistema operativo y dependencias base
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copia de archivos de requerimientos e instalación
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia del código fuente y directorio de datos
COPY app.py .
# El directorio de datos se montará como volumen, pero se crea la estructura
RUN mkdir -p data

# Exposición del puerto estándar de Streamlit
EXPOSE 8501

# Configuración de ejecución
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]