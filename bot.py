from dotenv import load_dotenv
from pyexiv2 import ImageData
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from numpy import load as np_load
from search_utils import extract_features, create_or_load_index
from os import path as os_path
from os import environ as os_environ
from os import remove as os_remove
from json import dump as json_dump
from json import load as json_load
from PIL import Image, ImageDraw
import subprocess
import bot_utilis

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


def generate_iphone_template(phone_name, cover_image_path, width_px, height_px):
    # Dimensioni del piano di stampa
    canvas_width = 3507
    canvas_height = 4960

    # Coordinate per posizionare l'immagine della cover
    start_x = 625
    start_y = 472

    # Distanza dal bordo sinistro per la linea nera
    black_line_x = 11

    # Carica l'immagine della cover e ridimensiona
    cover_image = Image.open(cover_image_path).convert("RGBA")
    cover_image = cover_image.rotate(270, expand=True)
    cover_image = cover_image.resize((width_px, height_px))

    # Crea un canvas trasparente
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    # Crea una linea nera
    draw = ImageDraw.Draw(canvas)
    line_y_start = start_y
    line_y_end = start_y + height_px
    draw.line([(black_line_x, line_y_start), (black_line_x, line_y_end)], fill="black", width=5)

    # Posiziona l'immagine della cover sul canvas
    canvas.paste(cover_image, (start_x, start_y), cover_image)

    filename = f"{phone_name}_template.png"
    canvas.save(filename, dpi=(11811, 11811))

    # Percorso dell'eseguibile exiftool incluso nel progetto
    exiftool_path = './exiftool'

    # Aggiungi metadati usando exiftool
    metadata_command = (
        f'{exiftool_path} -PixelsPerUnitX=11811 -PixelsPerUnitY=11811 '
        f'-ResolutionUnit=meters -SRGBRendering=Perceptual '
        f'-Gamma=2.2 {filename}'
    )

    try:
        subprocess.run(metadata_command, shell=True, check=True)
        print(f"Metadati aggiunti a {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'aggiunta dei metadati: {e}")

    print(f"Template salvato come {filename}")
    return filename


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
        [InlineKeyboardButton("Genera Template", callback_data='generate_template')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    with open(matched_image_path, 'rb') as image_file:
        await update.message.reply_photo(photo=image_file, reply_markup=reply_markup)


from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import logging


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if user_id not in user_search_states:
        await query.answer("Errore: Stato di ricerca non trovato.", show_alert=True)
        return

    user_state = user_search_states[user_id]

    if query.data == 'incorrect':
        # Gestisci la risposta "Non corretto"
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
        await query.edit_message_caption("Grazie per aver confermato l'immagine! 😊")

    elif query.data == 'generate_template':
        # Mostra il menu di scelta della serie di iPhone
        keyboard = bot_utilis.create_series_menu()
        await query.message.reply_text("Scegli la serie di iPhone:", reply_markup=keyboard)
        await query.answer()

    elif query.data.startswith('show_ip'):
        # Mostra il menu di modelli per la serie selezionata
        series_menu = {
            'show_ip15_series': bot_utilis.create_ip15_models_menu(),
            'show_ip14_series': bot_utilis.create_ip14_models_menu(),
            'show_ip13_series': bot_utilis.create_ip13_models_menu(),
            'show_ip12_series': bot_utilis.create_ip12_models_menu(),
            'show_ip11_series': bot_utilis.create_ip11_models_menu()
        }
        keyboard = series_menu.get(query.data, bot_utilis.create_series_menu())
        await query.message.edit_text("Scegli il modello di iPhone:", reply_markup=keyboard)
        await query.answer()

    elif query.data.startswith('ip'):
        phone_name = query.data

        try:
            matched_image_path = image_paths[user_state["labels"][user_state["current_index"]]]
        except IndexError:
            await query.answer("Errore: Impossibile trovare l'immagine corrispondente.", show_alert=True)
            return

        try:
            # Carica i dati dal JSON
            with open("misure_modelli.json", 'r') as file:
                model_data = json_load(file)
            logging.info(f"Caricato JSON")
        except Exception as e:
            logging.error(f"Errore nel caricamento del JSON: {e}")
            await query.answer("Errore: Impossibile caricare i dati del modello.", show_alert=True)
            return

        try:
            if phone_name in model_data["modelli"]:
                width_px = model_data["modelli"][phone_name]["width"]
                height_px = model_data["modelli"][phone_name]["height"]

                logging.info(f"Generazione template per {phone_name} con dimensioni {width_px}x{height_px}")
                await query.message.edit_text(f"Generazione template per {phone_name}...")

                template_path = generate_iphone_template(phone_name, matched_image_path, width_px, height_px)

                # Invia il template generato all'utente
                with open(template_path, 'rb') as template_file:
                    await query.message.reply_document(document=template_file, caption=f"Ecco il template per {phone_name}!")
                    # Cancella il template temporaneo
                    os_remove(template_path)
                    template_path = template_path +"_original"
                    os_remove(template_path)
                await query.answer()

            else:
                await query.answer("Modello di iPhone non riconosciuto. Riprova.", show_alert=True)

        except Exception as e:
            logging.error(f"Errore durante la generazione del template: {e}")
            await query.answer("Errore durante la generazione del template.", show_alert=True)

    elif query.data == 'back_to_series_menu':
        # Torna al menu delle serie di iPhone
        keyboard = bot_utilis.create_series_menu()
        await query.message.edit_text("Scegli la serie di iPhone:", reply_markup=keyboard)
        await query.answer()

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