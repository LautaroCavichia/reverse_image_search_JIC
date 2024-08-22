from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import png


def create_series_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 15 Series", callback_data='show_ip15_series')],
        [InlineKeyboardButton("iPhone 14 Series", callback_data='show_ip14_series')],
        [InlineKeyboardButton("iPhone 13 Series", callback_data='show_ip13_series')],
        [InlineKeyboardButton("iPhone 12 Series", callback_data='show_ip12_series')],
        [InlineKeyboardButton("iPhone 11 Series", callback_data='show_ip11_series')]
    ]
    return InlineKeyboardMarkup(keyboard)

# Bottoni per i modelli specifici di una serie di iPhone
def create_ip15_models_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 15", callback_data='ip15')],
        [InlineKeyboardButton("iPhone 15 Plus", callback_data='ip15_plus')],
        [InlineKeyboardButton("iPhone 15 Pro", callback_data='ip15_pro')],
        [InlineKeyboardButton("iPhone 15 Pro Max", callback_data='ip15_promax')],
        [InlineKeyboardButton("Indietro", callback_data='back_to_series_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_ip14_models_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 14", callback_data='ip14')],
        [InlineKeyboardButton("iPhone 14 Plus", callback_data='ip14_plus')],
        [InlineKeyboardButton("iPhone 14 Pro", callback_data='ip14_pro')],
        [InlineKeyboardButton("iPhone 14 Pro Max", callback_data='ip14_promax')],
        [InlineKeyboardButton("Indietro", callback_data='back_to_series_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_ip13_models_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 13", callback_data='ip13')],
        [InlineKeyboardButton("iPhone 13 Mini", callback_data='ip13_mini')],
        [InlineKeyboardButton("iPhone 13 Pro", callback_data='ip13_pro')],
        [InlineKeyboardButton("iPhone 13 Pro Max", callback_data='ip13_promax')],
        [InlineKeyboardButton("Indietro", callback_data='back_to_series_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_ip12_models_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 12", callback_data='ip12')],
        [InlineKeyboardButton("iPhone 12 Mini", callback_data='ip12_mini')],
        [InlineKeyboardButton("iPhone 12 Pro Max", callback_data='ip12_promax')],
        [InlineKeyboardButton("Indietro", callback_data='back_to_series_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_ip11_models_menu():
    keyboard = [
        [InlineKeyboardButton("iPhone 11", callback_data='ip11')],
        [InlineKeyboardButton("iPhone 11 Pro", callback_data='ip11_pro')],
        [InlineKeyboardButton("iPhone 11 Pro Max", callback_data='ip11_promax')],
        [InlineKeyboardButton("Indietro", callback_data='back_to_series_menu')]
    ]
    return InlineKeyboardMarkup(keyboard)

