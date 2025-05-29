# 📚 NLP Projesi: Dini Metin Yorumlarında Benzerlik Öneri Sistemi

Bu projede, dini metinlerden (özellikle Reddit üzerindeki yorumlar) elde edilen veriler üzerinde doğal dil işleme (NLP) yöntemleriyle metinler arası benzerlik ölçülerek öneri sistemi geliştirildi. Hem TF-IDF hem de Word2Vec yöntemleriyle çalışıldı.

## 🔍 Amaç
Verilen bir dini içerikli metne benzer yorumları bulmak için farklı vektörleme ve benzerlik ölçüm tekniklerini kullanmak.

## 📂 Proje Yapısı
```
.
├── religious_texts.csv                  # Ham veri
├── clean_lemmatized.csv                # Lemmatize edilmiş veri
├── clean_stemmed.csv                   # Stem edilmiş veri
├── tfidf_lemmatized.csv                # TF-IDF matrisleri
├── tfidf_stemmed.csv
├── word2vec_models/                    # 16 Word2Vec modeli
├── model_outputs.csv                   # Giriş yorumu için model çıktıları
├── model_average_similarity_scores.csv
├── manual_scoring_filled.csv           # Elle verilmiş puanlar
├── model_comparison_scores.csv         # Otomatik + manuel puanların karşılaştırması
├── jaccard_matrix.csv                  # Öneri kümeleri arası Jaccard skorları
├── scripts/
│   ├── word2vec.py                     # Model eğitimi
│   ├── tfidf_vectorizer.py             # TF-IDF üretimi
│   ├── lemmastem.py                    # Ön işleme
│   ├── generate_model_outputs.py       # Model çıktıları üretir
│   ├── compute_average_similarity.py   # Ortalama cosine skorları
│   ├── compare_models_scores.py        # Model karşılaştırma tablosu
│   ├── jaccard.py                      # Jaccard matrisi üretimi
```

## ⚙ Kullanılan Kütüphaneler
- `nltk`
- `gensim`
- `scikit-learn`
- `pandas`
- `matplotlib`

## 📌 Çalıştırma Sırası
1. **Veri Temizleme:**
```bash
python lemmastem.py
```
2. **TF-IDF Matrisi Oluşturma:**
```bash
python tfidf_vectorizer.py
```
3. **Word2Vec Eğitim:**
```bash
python word2vec.py
```
4. **Öneri Üretimi:**
```bash
python generate_model_outputs.py
```
5. **Cosine Ortalama Skor:**
```bash
python compute_average_similarity.py
```
6. **Manuel Skorlarla Karşılaştırma:**
```bash
python compare_models_scores.py
```
7. **Jaccard Benzerlik Matrisi:**
```bash
python jaccard.py
```

## 🧠 Kullanılan Model Parametreleri
- `Word2Vec`: CBOW vs SkipGram
- `window`: 2 / 4
- `vector_size`: 100 / 300
- `preprocessing`: Lemmatized / Stemmed

16 kombinasyonun her biri ayrı model olarak eğitildi.

## 🧾 Örnek Giriş
```bash
Input: observe lying vanity forsake mercy
```
> Sistem tüm modeller üzerinden giriş yorumuna en benzer 5 yorumu üretir.

## 📈 Rapor
Tüm açıklamalar için lütfen [final raporu](./RAPOR.md) inceleyiniz.

## 🔗 GitHub
https://github.com/Mahsumx/NLPodev
