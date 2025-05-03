import pandas as pd
from gensim.models import Word2Vec
import os

# Dosyaları yükle
lemma_df = pd.read_csv("clean_lemmatized.csv")
stem_df = pd.read_csv("clean_stemmed.csv")

# Giriş verileri: boş olmayanları listele
lemma_corpus = [text.split() for text in lemma_df["lemmatized"].dropna()]
stem_corpus = [text.split() for text in stem_df["stemmed"].dropna()]

# Parametre seçenekleri
datasets = [("lemmatized", lemma_corpus), ("stemmed", stem_corpus)]
sg_values = [0, 1]  # CBOW: 0, Skip-Gram: 1
window_sizes = [2, 4]
dimensions = [100, 300]

# Model klasörü
os.makedirs("word2vec_models", exist_ok=True)

# 16 modeli üret
for name, corpus in datasets:
    for sg in sg_values:
        for window in window_sizes:
            for dim in dimensions:
                model = Word2Vec(sentences=corpus, vector_size=dim, window=window, sg=sg, min_count=2, workers=4)
                method = "cbow" if sg == 0 else "skipgram"
                filename = f"word2vec_models/word2vec_{name}_{method}_window{window}_dim{dim}.model"
                model.save(filename)
                print(f"✅ Model kaydedildi: {filename}")
