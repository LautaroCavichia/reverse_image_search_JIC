# Utilizza una immagine base di Python con supporto per pip
FROM python:3.10-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia i file requirements.txt nel container
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Disabilita l'uso di CUDA e ottimizzazioni oneDNN per TensorFlow
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS="0"

# Copia tutto il contenuto della directory corrente nel container
COPY . .

# Specifica il comando per eseguire il tuo script Python
CMD ["python", "run_all.py"]