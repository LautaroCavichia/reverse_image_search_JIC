from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from numpy import load as np_load
from search_utils import extract_features, create_or_load_index
from os import path as os_path
from os import environ as os_environ
from json import dump as json_dump
from json import load as json_load

load_dotenv()

TELEGRAM_TOKEN = os_environ.get("TELEGRAM_TOKEN")
TEMP_IMAGE_PATH = "temp.jpg"

features = np_load("features.npy")
image_paths = np_load("image_paths.npy")
hnsw_index = create_or_load_index(features.shape[1], features, "image_index.bin")

STATS_FILE_PATH = "stats.json"


def load_stats():
    if os_path.exists(STATS_FILE_PATH):
        with open(STATS_FILE_PATH, 'r') as f:
            return json_load(f)
    return {"total_requests": 0, "incorrect": 0}


def save_stats(stats):
    with open(STATS_FILE_PATH, 'w') as f:
        json_dump(stats, f, indent=2)


stats = load_stats()

user_search_states = {}


async def handle_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for new_user in update.message.new_chat_members:
        welcome_message = (
            f"Benvenuto {new_user.full_name} a CaseLink!\n\n"
            "Ecco come funziona:\n"
            "1. Inviami una foto della cover con la grafica che vuoi (ritagliata a solo la cover) e cercherò quella più simile nel nostro database.\n"
            "2. Dopo aver inviato un'immagine, riceverai un suggerimento con un'immagine simile.\n"
            "3. Puoi indicare se l'immagine è corretta o meno e potrai chiedere una nuova immagine.\n\n"
        )
        welcome_image_path = "Welcome.jpg"


        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=open(welcome_image_path, 'rb'),
            caption=welcome_message
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = (
        "Inviami una foto della cover con la grafica che vuoi (ritagliata) e cercherò la grafica originale nel nostro database.\n"
        "Se l'immagine non è corretta, puoi chiedere una nuova immagine premendo il pulsante apposito.\n"
    )
    welcome_image_path = "Welcome.jpg"
    si_no_image_path = "si_no.jpg"

    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(welcome_image_path, 'rb'),
    )
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=open(si_no_image_path, 'rb'),
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

    stats["total_requests"] += 1
    save_stats(stats)

    await send_image_result(update, context, user_id)


async def send_image_result(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id):
    user_state = user_search_states[user_id]
    current_label = user_state["labels"][user_state["current_index"]]
    matched_image_path = image_paths[current_label]

    keyboard = [
        [InlineKeyboardButton("👍 Corretta!", callback_data='correct')],
        [InlineKeyboardButton("👎 Non corretta, mostrami un'altra", callback_data='incorrect')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    with open(matched_image_path, 'rb') as image_file:
        await update.message.reply_photo(photo=image_file, reply_markup=reply_markup)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    user_state = user_search_states[user_id]

    if query.data == 'incorrect':
        stats["incorrect"] += 1
        stats["total_requests"] += 1
        save_stats(stats)
        user_state["current_index"] += 1

        if user_state["current_index"] < len(user_state["labels"]):
            await query.answer("Provo un'altra immagine...")
            await send_image_result(query, context, user_id)
        else:
            await query.edit_message_caption("Mi dispiace, non ci sono altre immagini disponibili, prova a ritagliare "
                                             "la foto. Oppure è possibile che non faccia parte del nostro database.")
            await query.answer("Non ci sono altre immagini da mostrare. 😞")

    elif query.data == 'correct':
        await query.answer("Grazie per aver confermato l'immagine! 😊")


async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response_message = (
        f"Statistiche del bot:\n"
        f"Totale richieste: {stats['total_requests']}\n"
        f"Immagini non trovate: {stats['incorrect']}\n"
    )
    await update.message.reply_text(response_message)


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_user))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stats", show_stats))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.run_polling()


if __name__ == "__main__":
    main()