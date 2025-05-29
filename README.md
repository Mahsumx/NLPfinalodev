# 📚 NLP Projesi: Dini Metin Yorumlarında Benzerlik Öneri Sistemi

Bu projede, dini metinlerden (özellikle Reddit üzerindeki yorumlar) elde edilen veriler üzerinde doğal dil işleme (NLP) yöntemleriyle metinler arası benzerlik ölçülerek öneri sistemi geliştirildi. Hem TF-IDF hem de Word2Vec yöntemleriyle çalışıldı.

## 🔍 Amaç
Verilen bir dini içerikli metne benzer yorumları bulmak için farklı vektörleme ve benzerlik ölçüm tekniklerini kullanmak.

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
