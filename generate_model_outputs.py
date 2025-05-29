import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

# Ayarlar
MODELS_DIR = "word2vec_models"
DATA_PATHS = [
    ("clean_lemmatized.csv", "lemmatized"),
    ("clean_stemmed.csv", "stemmed")
]
OUTPUT_CSV = "model_outputs.csv"

# Çıktılar
output_rows = []

# Her veri kümesi için işlem
for data_path, text_column in DATA_PATHS:
    print(f"\n📂 Veri işleniyor: {data_path} ({text_column})")
    df = pd.read_csv(data_path)
    df[text_column] = df[text_column].astype(str).fillna("")

    total_models = [m for m in os.listdir(MODELS_DIR) if m.endswith(".model") and text_column in m]
    print(f"🔍 {len(total_models)} model bulundu ({text_column})")

    for model_file in total_models:
        print(f"⚡ Model işleniyor: {model_file}")
        model_path = os.path.join(MODELS_DIR, model_file)
        model = Word2Vec.load(model_path)

        # Modele göre giriş kelimesi
        if "stemmed" in model_file:
            input_tokens = word_tokenize("merc forgiv")
        else:
            input_tokens = word_tokenize("mercy forgiveness")

        filtered = [t for t in input_tokens if t in model.wv.key_to_index]
        if not filtered:
            print("⛔ Giriş kelimeleri modelde yok.")
            continue

        input_vec = np.mean([model.wv[t] for t in filtered], axis=0).reshape(1, -1)

        similarities = []
        for text in df[text_column]:
            tokens = [t for t in word_tokenize(text) if t in model.wv.key_to_index]
            if not tokens:
                similarities.append(0)
                continue
            vec = np.mean([model.wv[t] for t in tokens], axis=0).reshape(1, -1)
            sim = cosine_similarity(input_vec, vec)[0][0]
            similarities.append(sim)

        df_temp = df.copy()
        df_temp["similarity"] = similarities
        df_top5 = df_temp.sort_values("similarity", ascending=False).head(5)

        for rank, row in enumerate(df_top5.itertuples(), start=1):
            output_rows.append({
                "model_name": model_file.replace(".model", ""),
                "suggestion_rank": rank,
                "suggestion_text": getattr(row, text_column),
                "similarity": row.similarity
            })

# CSV kaydet
pd.DataFrame(output_rows).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Tüm model çıktıları kaydedildi: {OUTPUT_CSV}")
