import cv2
import numpy as np
import hnswlib
import os
import logging
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import torch

# Impostazioni per usare solo la CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configurazione di logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica il modello EfficientNet pre-addestrato
model_name = 'efficientnet-b0'
model = EfficientNet.from_pretrained(model_name)
model.eval()  # Imposta il modello in modalità di valutazione

# Trasformazioni delle immagini
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    if image is None:
        logger.error(f"Errore nel caricamento dell'immagine: {image_path}")
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Aggiungi una dimensione batch
    return image

def extract_features(image_path):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        features = model(image)
    # Normalizza le feature
    features = features.numpy().flatten()
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