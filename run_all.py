import subprocess

def run_flask():
    subprocess.run(["python", "app.py"])

def run_bot():
    subprocess.run(["python", "bot.py"])

if __name__ == "__main__":
    # Avvia Flask in un processo separato
    flask_process = subprocess.Popen(["python", "app.py"])

    # Avvia il bot Telegram in un altro processo separato
    bot_process = subprocess.Popen(["python", "bot.py"])

    flask_process.wait()
    bot_process.wait()