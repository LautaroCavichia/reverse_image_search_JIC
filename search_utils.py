import cv2
import numpy as np
import hnswlib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica il modello VGG16 pre-addestrato e rimuovi l'ultimo layer
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


def load_and_preprocess_image(image_path):
    """Carica e preprocessa un'immagine."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Errore nel caricamento dell'immagine: {image_path}")
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def extract_features(image_path):
    """Estrae le feature dall'immagine utilizzando il modello VGG16."""
    image = load_and_preprocess_image(image_path)
    features = model.predict(image)
    # Normalizza le feature
    features = features.flatten()
    features /= np.linalg.norm(features)
    return features


def create_or_load_index(dim, features, index_file):
    """Crea o carica un indice HNSW."""
    hnsw_index = hnswlib.Index(space='l2', dim=dim)
    if os.path.exists(index_file):
        logger.info(f"Caricamento dell'indice da {index_file}")
        hnsw_index.load_index(index_file)
    else:
        logger.info("Creazione di un nuovo indice")
        hnsw_index.init_index(max_elements=len(features), ef_construction=200, M=16)
        hnsw_index.add_items(features, np.arange(len(features)))
        hnsw_index.save_index(index_file)
        logger.info(f"Indice salvato in {index_file}")

    # Imposta un valore di `ef` più alto per migliorare la precisione della ricerca
    hnsw_index.set_ef(100)
    return hnsw_index