# 🎓 Kariyer Öneri Uzman Sistemi

Bu proje, **RIASEC (Holland Codes)** modeline dayalı, kural tabanlı ve makine öğrenmesi destekli hibrit bir kariyer öneri sistemidir. Kullanıcının kişilik özellikleri ve ilgi alanları doğrultusunda en uygun meslek önerilerini sunmayı amaçlar.

## 🚀 Proje Özellikleri

- Kullanıcıdan **ilgi ve tercihleri** ile ilgili sorular alır
- **Kural tabanlı çıkarım (IF–THEN kuralları)** ile öneriler sunar
- **Makine öğrenmesi modeli** ile veriye dayalı öneriler üretir
- **RIASEC profili** oluşturur ve buna dayalı meslek önerileri yapar
- Kullanıcı geri bildirimleriyle öneri doğruluğunu artırabilir (planlanan)

## 📦 Kullanılan Veri Setleri

**Karar Verilecek**

## 🏗️ Planlanan Proje Yapısı

```
/src
  ├── main.py
  ├── rules.json
  ├── model.pkl
/tests
  ├── test_input.json
README.md
requirements.txt
```

## 🧠 Kullanılacak Teknolojiler

- Python 3.x
- scikit-learn
- pandas
- JSON (kural tabanı formatı)

## 🔍 Çalıştırmak için

1. Dependency yükleyin:

   ```
   pip install -r requirements.txt
   ```

2. Sistemi çalıştırın:

   ```
   python main.py
   ```

3. Gelen soruları yanıtlayın -> önerilen meslekleri alın.
