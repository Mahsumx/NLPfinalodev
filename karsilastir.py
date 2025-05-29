import pandas as pd

# CSV dosyasını yükle
df = pd.read_csv("manual_scoring_filled_normalized.csv")

# Model bazında ortalama cosine similarity ve manuel puanları hesapla
comparison_df = df.groupby("model_name")[["similarity", "manual_score"]].mean().reset_index()
comparison_df.columns = ["model_name", "avg_cosine_similarity", "avg_manual_score"]

# Sonucu CSV dosyasına yaz
comparison_df.to_csv("model_comparison_scores.csv", index=False)

print("✅ Model karşılaştırma tablosu başarıyla oluşturuldu.")
