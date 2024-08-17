from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import numpy as np
from search_utils import extract_features, create_or_load_index
import os

load_dotenv()
# Carica le variabili d'ambiente da un file .env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TEMP_IMAGE_PATH = "temp.jpg"

features = np.load("features.npy")
image_paths = np.load("image_paths.npy")
hnsw_index = create_or_load_index(features.shape[1], features, "image_index.bin")

# Dizionario per memorizzare lo stato della ricerca degli utenti
user_search_states = {}


async def handle_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for new_user in update.message.new_chat_members:
        welcome_message = (
            f"Benvenuto {new_user.full_name} nella chat! 😊\n\n"
            "Ecco come funziona:\n"
            "1. Inviami una foto della cover con la grafica che vuoi (ritagliata a solo la cover) e cercherò quella più simile nel nostro database.\n"
            "2. Dopo aver inviato un'immagine, riceverai un suggerimento con un'immagine simile.\n"
            "3. Puoi indicare se l'immagine è corretta o meno utilizzando i pulsanti 👍 o 👎.\n\n"
        )
        welcome_image_path = "welcome_image.jpg"

        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=open(welcome_image_path, 'rb'),
            caption=welcome_message
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "Ecco come funziona:\n"
        "1. Inviami una foto della cover con la grafica che vuoi (ritagliata a solo la cover) e cercherò quella più simile nel nostro database.\n"
        "2. Dopo aver inviato un'immagine, riceverai un suggerimento con un'immagine simile.\n"
        "3. Puoi indicare se l'immagine è corretta o meno utilizzando i pulsanti 👍 o 👎.\n\n"
    )
    welcome_image_path = "welcome_image.jpg"

    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(welcome_image_path, 'rb'),
        caption=welcome_message
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(TEMP_IMAGE_PATH)

    query_features = extract_features(TEMP_IMAGE_PATH)
    labels, distances = hnsw_index.knn_query(query_features, k=5)

    user_id = update.message.from_user.id
    user_search_states[user_id] = {
        "labels": labels[0],
        "distances": distances[0],
        "current_index": 0
    }

    await send_image_result(update, context, user_id)


async def send_image_result(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id):
    user_state = user_search_states[user_id]
    current_label = user_state["labels"][user_state["current_index"]]
    matched_image_path = image_paths[current_label]

    keyboard = [
        [InlineKeyboardButton("👍 Corretto", callback_data='correct')],
        [InlineKeyboardButton("👎 Non corretto", callback_data='incorrect')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    with open(matched_image_path, 'rb') as image_file:
        await update.message.reply_photo(photo=image_file, reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    user_state = user_search_states[user_id]

    if query.data == 'correct':
        await query.answer("Grazie per il feedback!")
        await query.edit_message_caption("Sono felice che tu abbia trovato l'immagine giusta! 😊")

    elif query.data == 'incorrect':
        user_state["current_index"] += 1

        if user_state["current_index"] < len(user_state["labels"]):
            await query.answer("Provo un'altra immagine...")
            await send_image_result(query, context, user_id)
        else:
            await query.answer("Mi dispiace, non ci sono altre immagini disponibili, prova a ritagliare la foto. "
                               "Oppure è possibile che non faccia parte del nostro database.")
            await query.edit_message_caption("Non ci sono altre immagini da mostrare. 😞")


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_user))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.run_polling()


if __name__ == "__main__":
    main()