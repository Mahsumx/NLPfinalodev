import pandas as pd

# Dosyayı yükle
df = pd.read_csv("model_outputs.csv")

# Model bazında ortalama cosine similarity hesapla
average_scores = df.groupby("model_name")["similarity"].mean().reset_index()
average_scores.columns = ["model_name", "average_similarity"]

# CSV olarak kaydet
average_scores.to_csv("model_average_similarity_scores.csv", index=False)

print("✅ Ortalama benzerlik skorları hesaplandı ve 'model_average_similarity_scores.csv' dosyasına kaydedildi.")
