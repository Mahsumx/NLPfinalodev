# Dini Metin Yorum Benzerliği – NLP Projesi

Bu projede, Kur’an ve İncil gibi kutsal metinler kullanılarak doğal dil işleme (NLP) teknikleriyle anlamsal benzerlik analizi yapılmıştır. Proje kapsamında vektör temsili, kelime benzerliği, frekans analizleri gibi yöntemlerle metinler arasında semantik benzerlikler ortaya konmuştur.

## 📁 Veri Setleri
- **Kur’an:** Sahih International İngilizce çeviri (tanzil.net)
- **İncil:** King James Version (KJV), İngilizce çeviri (Kaggle ve Bible API)

## 🧹 Veri Temizleme Adımları
- Küçük harfe çevirme
- Noktalama işaretlerini kaldırma
- Tokenization (nltk)
- Stopword temizliği
- Lemmatizasyon (WordNetLemmatizer)
- Stemming (PorterStemmer)

## 📊 Analizler
### 1. Zipf Analizi
Ham, lemmatized ve stemmed metinler için kelime frekansı log-log düzleminde incelendi.

### 2. TF-IDF Vektörizasyonu
- `tfidf_lemmatized.csv` ve `tfidf_stemmed.csv` olarak kaydedildi.
- Her cümlenin kelime ağırlıkları çıkarıldı.

### 3. Word2Vec Modelleri
- Toplam **16 model** üretildi:
  - 2 veri türü (lemmatized / stemmed)
  - 2 algoritma (CBOW / Skip-Gram)
  - 2 pencere boyutu (2 / 4)
  - 2 vektör boyutu (100 / 300)
- Her model `.model` olarak kaydedildi ve açıklaması `.txt` dosyasına yazıldı.

## 🧠 Örnek Çıktı
```text
Model: word2vec_lemmatized_cbow_window2_dim100.model
Test Kelime: "love"
Benzer Kelimeler:
- remember: 0.942
- doth: 0.937
- desire: 0.931
- merciful: 0.931
- hope: 0.927
...
```

## 🚀 Nasıl Çalıştırılır?
1. Python 3.10+ sürümü kurulu olmalıdır.
2. Gerekli kütüphaneler:
```
pip install pandas nltk gensim matplotlib scikit-learn
```
3. Modelleri eğitmek için:
```
python train_16_models.py
```
4. Örnek kelime sorgulamak için:
```
python recommend.py
```