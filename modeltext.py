import os

# Ayarlar
types = ["lemmatized", "stemmed"]
sg_map = {0: "CBOW", 1: "Skip-Gram"}
window_sizes = [2, 4]
dimensions = [100, 300]

# Açıklama klasörü
desc_folder = "word2vec_models"

# Dosyaları oluştur
for data_type in types:
    for sg in [0, 1]:
        for window in window_sizes:
            for dim in dimensions:
                method = "cbow" if sg == 0 else "skipgram"
                file_base = f"word2vec_{data_type}_{method}_window{window}_dim{dim}"
                model_file = os.path.join(desc_folder, file_base + ".model")
                desc_file = os.path.join(desc_folder, file_base + ".txt")
                
                description = (
                    f"Veri tipi: {data_type}\n"
                    f"Model türü: {sg_map[sg]}\n"
                    f"Pencere boyutu: {window}\n"
                    f"Vektör boyutu: {dim}\n"
                    f"Açıklama: Bu model {data_type} metinler üzerinde {sg_map[sg]} algoritması ile eğitilmiştir. "
                    f"Pencere boyutu {window} ve her kelime {dim} boyutlu vektöre gömülmüştür."
                )
                
                with open(desc_file, "w", encoding="utf-8") as f:
                    f.write(description)
                
                print(f"📄 Açıklama yazıldı: {desc_file}")
