import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Gerekli NLTK verileri indir (bir kez çalıştırmak yeterli)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Veri dosyasını oku
df = pd.read_csv("religious_texts.csv")

# İngilizce stopwords listesi
stop_words = set(stopwords.words("english"))

# Temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # sadece harfleri bırak
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Lemmatizer ve Stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

lemmatized_texts = []
stemmed_texts = []

# Her cümleyi işle
for text in df["text"]:
    tokens = clean_text(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed = [stemmer.stem(token) for token in tokens]
    
    lemmatized_texts.append(" ".join(lemmatized))
    stemmed_texts.append(" ".join(stemmed))

# Yeni sütunları ekle
df["lemmatized"] = lemmatized_texts
df["stemmed"] = stemmed_texts

# CSV olarak kaydet (istersen)
df[["source", "lemmatized"]].to_csv("clean_lemmatized.csv", index=False)
df[["source", "stemmed"]].to_csv("clean_stemmed.csv", index=False)
