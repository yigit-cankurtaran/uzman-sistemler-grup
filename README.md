# 🎓 Kariyer Öneri Uzman Sistemi

Bu proje, **RIASEC (Holland Codes)** modeline dayalı, kural tabanlı ve makine öğrenmesi destekli hibrit bir kariyer öneri sistemidir. Kullanıcının kişilik özellikleri ve ilgi alanları doğrultusunda en uygun meslek önerilerini sunmayı amaçlar.

## 🚀 Proje Özellikleri

- Kullanıcıdan **ilgi ve tercihleri** ile ilgili sorular alır
- **Makine öğrenmesi modeli** ile öneriler sunar
- **RIASEC profili** oluşturur ve buna dayalı meslek önerileri yapar

## 🧠 Kullanılan Teknolojiler

- Python 3.x
- pytorch
- pandas
- scikit-learn
- JSON (cevaplar formatı)

## 🔍 Çalıştırmak için

1. Dependencyleri yükleyin.

2. Sistemi çalıştırın:

   ```
   python main.py
   ```

3. Gelen soruları yanıtlayın.

4. Modeli eğitin
```
python training.py --data_path=dataset.csv --epochs=200 --patience=20
```

5. Modeli cevaplar üzerinde çalıştırın
```
python inference.py --json_path cevaplar.json
```
