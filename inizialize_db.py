import os
import numpy as np
from search_utils import extract_features

IMAGE_DIR = "image_db"
FEATURE_FILE = "features.npy"
IMAGE_PATHS_FILE = "image_paths.npy"

image_paths = [
    os.path.join(IMAGE_DIR, fname)
    for fname in os.listdir(IMAGE_DIR)
    if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
]

features = []
for image_path in image_paths:
    print(f"Estrazione caratteristiche da {image_path}")
    features.append(extract_features(image_path))


features = np.array(features)

np.save(FEATURE_FILE, features)
np.save(IMAGE_PATHS_FILE, image_paths)

print(f"Features salvate in {FEATURE_FILE}")
print(f"Percorsi delle immagini salvati in {IMAGE_PATHS_FILE}")