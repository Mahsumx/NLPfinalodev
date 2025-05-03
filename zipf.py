import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from collections import Counter

# Temizleme fonksiyonu
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return nltk.word_tokenize(text)

# Zipf çizim fonksiyonu
def plot_zipf(text_series, title, filename):
    all_text = " ".join(text_series.astype(str))
    tokens = tokenize(all_text)
    freq_dist = Counter(tokens)
    most_common = freq_dist.most_common(500)
    
    ranks = list(range(1, len(most_common)+1))
    frequencies = [freq for word, freq in most_common]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies)
    plt.xlabel("Kelime Sırası (log)")
    plt.ylabel("Frekans (log)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Veriyi yükle
df_ham = pd.read_csv("religious_texts.csv")
df_lemma = pd.read_csv("clean_lemmatized.csv")
df_stem = pd.read_csv("clean_stemmed.csv")

# Her biri için grafik oluştur
plot_zipf(df_ham["text"], "Zipf Analizi - Ham Metin", "zipf_ham_veri.png")
plot_zipf(df_lemma["lemmatized"], "Zipf Analizi - Lemmatized Metin", "zipf_lemmatized_veri.png")
plot_zipf(df_stem["stemmed"], "Zipf Analizi - Stemmed Metin", "zipf_stemmed_veri.png")
