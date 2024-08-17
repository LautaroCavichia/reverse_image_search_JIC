import cv2
import numpy as np
import hnswlib
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import logging

tf.config.set_visible_devices([], 'GPU')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Errore nel caricamento dell'immagine: {image_path}")
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def extract_features(image_path):
    image = load_and_preprocess_image(image_path)
    features = model.predict(image)
    # Normalizza le feature
    features = features.flatten()
    features /= np.linalg.norm(features)
    return features


def create_or_load_index(dim, features, index_file):
    hnsw_index = hnswlib.Index(space='l2', dim=dim)
    if os.path.exists(index_file):
        logger.info(f"Caricamento dell'indice da {index_file}")
        hnsw_index.load_index(index_file)
    else:
        logger.info("Creazione di un nuovo indice")
        hnsw_index.init_index(max_elements=len(features), ef_construction=400, M=40)
        hnsw_index.add_items(features, np.arange(len(features)))
        hnsw_index.save_index(index_file)
        logger.info(f"Indice salvato in {index_file}")

    hnsw_index.set_ef(200)
    return hnsw_index