import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Dosya yolları
lemmatized_path = "clean_lemmatized.csv"
stemmed_path = "clean_stemmed.csv"

# Verileri oku
df_lemma = pd.read_csv(lemmatized_path)
df_stem = pd.read_csv(stemmed_path)

# TF-IDF için maksimum kelime sayısı
max_features = 2000  # Bellek dostu olsun diye düşürüldü

# Lemmatized TF-IDF
vectorizer_lemma = TfidfVectorizer(max_features=max_features)
X_lemma = vectorizer_lemma.fit_transform(df_lemma["lemmatized"].astype(str))
df_tfidf_lemma = pd.DataFrame.sparse.from_spmatrix(X_lemma, columns=vectorizer_lemma.get_feature_names_out())
df_tfidf_lemma.to_csv("tfidf_lemmatized.csv", index=False)

# Stemmed TF-IDF
vectorizer_stem = TfidfVectorizer(max_features=max_features)
X_stem = vectorizer_stem.fit_transform(df_stem["stemmed"].astype(str))
df_tfidf_stem = pd.DataFrame.sparse.from_spmatrix(X_stem, columns=vectorizer_stem.get_feature_names_out())
df_tfidf_stem.to_csv("tfidf_stemmed.csv", index=False)

print("✅ TF-IDF çıktıları başarıyla oluşturuldu.")
