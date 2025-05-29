from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

# 🔧 Sabit yollar (sana özel ayarlandı)
model_path = "word2vec_models/word2vec_lemmatized_cbow_window2_dim100.model"
data_path = "clean_lemmatized.csv"
text_column = "lemmatized"

# Model ve veri yükle
model = Word2Vec.load(model_path)
df = pd.read_csv(data_path)

# Veriyi hazırla
df[text_column] = df[text_column].astype(str).fillna("")

# 🎯 Giriş cümlesi
input_text = "mercy and forgiveness".lower()
input_tokens = word_tokenize(input_text)
filtered_tokens = [t for t in input_tokens if t in model.wv.key_to_index]

if not filtered_tokens:
    print("⛔ Girilen kelimeler modelde bulunamadı.")
    exit()

input_vector = np.mean([model.wv[t] for t in filtered_tokens], axis=0).reshape(1, -1)

# Cosine similarity hesapla
similarities = []
for text in df[text_column]:
    tokens = [t for t in word_tokenize(str(text)) if t in model.wv.key_to_index]
    if not tokens:
        similarities.append(0)
        continue
    vec = np.mean([model.wv[t] for t in tokens], axis=0).reshape(1, -1)
    score = cosine_similarity(input_vector, vec)[0][0]
    similarities.append(score)

df["similarity"] = similarities
df_sorted = df.sort_values("similarity", ascending=False)

# ✅ En iyi 5 sonucu göster
print("\n📌 En Benzer 5 Sonuç:")
for i, row in df_sorted.head(5).iterrows():
    print(f"\n🔹 Benzerlik: {row['similarity']:.4f}")
    print(f"📖 Metin: {row[text_column]}")
