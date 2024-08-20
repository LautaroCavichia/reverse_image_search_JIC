from numpy import arange as np_arange
from numpy import linalg as np_linalg
import hnswlib
from os import path as os_path
from os import environ as os_environ
from logging import getLogger as logging_getLogger
from logging import basicConfig as logging_basicConfig
from logging import INFO as logging_INFO
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from torch import no_grad as torch_no_grad

# Disabilita l'utilizzo della GPU
os_environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os_environ["CUDA_VISIBLE_DEVICES"] = "-1"
os_environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


logging_basicConfig(level=logging_INFO)
logger = logging_getLogger(__name__)


model_name = 'efficientnet-b0'
model = EfficientNet.from_pretrained(model_name)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        logger.error(f"Errore nel caricamento dell'immagine: {image_path}, {e}")
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")


def extract_features(image_path):
    image = load_and_preprocess_image(image_path)
    with torch_no_grad():
        features = model(image)

    features = features.numpy().flatten()
    features /= np_linalg.norm(features)
    return features


def create_or_load_index(dim, features, index_file):
    hnsw_index = hnswlib.Index(space='l2', dim=dim)
    if os_path.exists(index_file):
        logger.info(f"Caricamento dell'indice da {index_file}")
        hnsw_index.load_index(index_file)
    else:
        logger.info("Creazione di un nuovo indice")
        hnsw_index.init_index(max_elements=len(features), ef_construction=450, M=45)
        hnsw_index.add_items(features, np_arange(len(features)))
        hnsw_index.save_index(index_file)
        logger.info(f"Indice salvato in {index_file}")

    hnsw_index.set_ef(200)
    return hnsw_index