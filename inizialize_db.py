import os
import numpy as np
from search_utils import extract_features

IMAGE_DIR = "image_db"
FEATURE_FILE = "features.npy"
IMAGE_PATHS_FILE = "image_paths.npy"

# Lista di tutti i percorsi delle immagini
image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith('.jpg') or fname.endswith('.png')]

# Estrai le caratteristiche da tutte le immagini
features = []
for image_path in image_paths:
    print(f"Estrazione caratteristiche da {image_path}")
    features.append(extract_features(image_path))

features = np.array(features)

# Salva le caratteristiche e i percorsi delle immagini in file .npy
np.save(FEATURE_FILE, features)
np.save(IMAGE_PATHS_FILE, image_paths)

print(f"Features salvate in {FEATURE_FILE}")
print(f"Percorsi delle immagini salvati in {IMAGE_PATHS_FILE}")