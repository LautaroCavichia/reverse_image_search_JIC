# Usa un'immagine base ufficiale di Python
FROM python:3.10-slim

# Imposta la variabile d'ambiente per evitare il buffering dell'output
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    wget \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# Scarica e installa ExifTool
RUN wget https://exiftool.org/exiftool-12.93.tar.gz -O /tmp/exiftool.tar.gz && \
    tar -xzf /tmp/exiftool.tar.gz -C /opt && \
    ln -s /opt/exiftool/exiftool /usr/local/bin/exiftool && \
    rm /tmp/exiftool.tar.gz

# Crea una directory per l'app e imposta come directory di lavoro
WORKDIR /app

# Copia il file requirements.txt e installa le dipendenze Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dell'app nel contenitore
COPY . /app/

# Comando per eseguire l'app
CMD ["python", "run_all.py"]