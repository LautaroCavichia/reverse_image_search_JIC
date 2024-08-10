from flask import Flask, request, jsonify
import numpy as np
from search_utils import extract_features, create_or_load_index
import os

# Configurazione Flask
app = Flask(__name__)

# Configurazione
IMAGE_DIR = "image_db"
FEATURE_FILE = "features.npy"
INDEX_FILE = "image_index.bin"
TEMP_IMAGE_PATH = "temp.jpg"

# Carica le caratteristiche e l'indice
if not os.path.exists(FEATURE_FILE):
    raise Exception(f"File {FEATURE_FILE} non trovato. Esegui lo script initialize_db.py per creare il database.")
features = np.load(FEATURE_FILE)
image_paths = np.load("image_paths.npy")
hnsw_index = create_or_load_index(features.shape[1], features, INDEX_FILE)

@app.route('/search', methods=['POST'])
def search_image():
    # Salva l'immagine temporaneamente
    file = request.files['file']
    file.save(TEMP_IMAGE_PATH)

    # Estrai le feature dall'immagine query
    query_features = extract_features(TEMP_IMAGE_PATH)

    # Normalizzazione delle feature
    query_features = query_features / np.linalg.norm(query_features)

    # Esegui la ricerca nell'indice con k=5 per avere più opzioni
    labels, distances = hnsw_index.knn_query(query_features, k=5)

    # Seleziona l'immagine più simile con una distanza inferiore a una soglia
    THRESHOLD = 0.5  # Questo valore può essere adattato in base al tuo dataset
    best_match = None

    for i, distance in enumerate(distances[0]):
        if distance < THRESHOLD:
            best_match = image_paths[labels[0][i]]
            break

    # Se nessun match è trovato sotto la soglia, restituisci il miglior risultato disponibile
    if best_match is None:
        best_match = image_paths[labels[0][0]]

    return jsonify({"matched_image": best_match})

if __name__ == '__main__':
    # Avvia Flask
    app.run(debug=True)