import cv2
import numpy as np
import hnswlib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import os

# Carica il modello VGG16 pre-addestrato e rimuovi l'ultimo layer
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def extract_features(image_path):
    image = load_and_preprocess_image(image_path)
    features = model.predict(image)
    return features.flatten()

def create_or_load_index(dim, features, index_file):
    hnsw_index = hnswlib.Index(space='l2', dim=dim)
    if os.path.exists(index_file):
        hnsw_index.load_index(index_file)
    else:
        hnsw_index.init_index(max_elements=len(features), ef_construction=200, M=16)
        hnsw_index.add_items(features, np.arange(len(features)))
        hnsw_index.save_index(index_file)
    return hnsw_index