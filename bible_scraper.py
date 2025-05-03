import csv
import time
import requests

# Ayet çekilecek kitaplar ve kaç tane ayet alınacağı
books = {
    "John": 100,
    "Matthew": 100,
    "Mark": 100,
    "Luke": 100,
    "Romans": 100
}

output_file = "bible_verses.csv"

with open(output_file, "w", newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["source", "text"])

    for book, count in books.items():
        for i in range(1, count + 1):
            verse_ref = f"{book} {i}:1"
            url = f"https://bible-api.com/{verse_ref}"
            try:
                response = requests.get(url)
                data = response.json()
                verse_text = data.get("text", "").replace("\n", " ").strip()
                if verse_text:
                    writer.writerow(["bible", verse_text])
                    print(f"{verse_ref} ✔")
                time.sleep(0.5)
            except Exception as e:
                print(f"Hata ({verse_ref}):", e)