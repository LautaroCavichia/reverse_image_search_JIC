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


if not os.path.exists(FEATURE_FILE):
    raise Exception(f"File {FEATURE_FILE} non trovato. Esegui lo script initialize_db.py per creare il database.")
features = np.load(FEATURE_FILE)
image_paths = np.load("image_paths.npy")
hnsw_index = create_or_load_index(features.shape[1], features, INDEX_FILE)

@app.route('/search', methods=['POST'])
def search_image():

    file = request.files['file']
    file.save(TEMP_IMAGE_PATH)

    query_features = extract_features(TEMP_IMAGE_PATH)

    query_features = query_features / np.linalg.norm(query_features)

    labels, distances = hnsw_index.knn_query(query_features, k=5)

    THRESHOLD = 0.5
    best_match = None

    for i, distance in enumerate(distances[0]):
        if distance < THRESHOLD:
            best_match = image_paths[labels[0][i]]
            break

    if best_match is None:
        best_match = image_paths[labels[0][0]]

    return jsonify({"matched_image": best_match})


if __name__ == '__main__':
    # Avvia Flask
    app.run(debug=True)