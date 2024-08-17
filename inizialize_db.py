import os
import numpy as np
from search_utils import extract_features

IMAGE_DIR = "image_db"
FEATURE_FILE = "features.npy"
IMAGE_PATHS_FILE = "image_paths.npy"

# Ottieni tutti i percorsi delle immagini supportate nella directory dell'immagine
image_paths = [
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
]

# Estrai le caratteristiche per ogni immagine e salva in un array
features = []
for image_path in image_paths:
    print(f"Estrazione caratteristiche da {image_path}")
    features.append(extract_features(image_path))

# Converti la lista delle caratteristiche in un array NumPy e salva i file
features = np.array(features)

np.save(FEATURE_FILE, features)
np.save(IMAGE_PATHS_FILE, image_paths)

print(f"Features salvate in {FEATURE_FILE}")
print(f"Percorsi delle immagini salvati in {IMAGE_PATHS_FILE}")