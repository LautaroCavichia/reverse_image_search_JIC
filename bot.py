from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from search_utils import extract_features, create_or_load_index
import numpy as np

file_path = "TELEGRAM_TOKEN.txt"
# Configurazione
def load_telegram_token(file_path):
    try:
        with open(file_path, "r") as f:
            token = f.read().strip()
            if not token:
                raise ValueError("Il file è vuoto.")
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' non trovato.")
    except ValueError as ve:
        raise ValueError(f"Errore nel contenuto del file: {ve}")
    except Exception as e:
        raise RuntimeError(f"Errore generico: {e}")


# Carica il token dal file
TELEGRAM_TOKEN = load_telegram_token("TELEGRAM_TOKEN.txt")
TEMP_IMAGE_PATH = "temp.jpg"

# Carica le caratteristiche e l'indice
features = np.load("features.npy")
image_paths = np.load("image_paths.npy")
hnsw_index = create_or_load_index(features.shape[1], features, "image_index.bin")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Ciao! Inviami un\'immagine e cercherò quella più simile nel database.')

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(TEMP_IMAGE_PATH)

    # Usa la funzione di ricerca locale
    query_features = extract_features(TEMP_IMAGE_PATH)
    labels, distances = hnsw_index.knn_query(query_features, k=1)
    matched_image_path = image_paths[labels[0][0]]

    # Invia l'immagine trovata all'utente
    with open(matched_image_path, 'rb') as image_file:
        await update.message.reply_photo(photo=image_file)

def main():
    # Configura il bot Telegram
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Avvia il bot
    application.run_polling()

if __name__ == "__main__":
    main()