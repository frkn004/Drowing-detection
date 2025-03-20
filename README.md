# Havuz Güvenlik Sistemi

Bu proje, yüzme havuzlarında boğulma vakalarını otomatik olarak tespit etmeyi amaçlayan bir bilgisayarlı görü sistemidir. Sistem beş ana modülden oluşur:

1. **Veri Toplama**: Video kaynaklarından görüntülerin alınması.
2. **İnsan Tespiti**: Havuz içindeki ve çevresindeki insanların tespit edilmesi.
3. **Hareket Analizi**: Yüzme hareketlerinin analiz edilmesi.
4. **Boğulma Tespiti**: Anormal hareket paternlerinin tanımlanması.
5. **Görselleştirme**: Sonuçların gösterilmesi ve uyarıların oluşturulması.

## Başlangıç

### Bağımlılıkları Yükleme

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

### Hazır Model ile Çalıştırma

Sistemin temel versiyonunu çalıştırmak için:

```bash
python src/main.py --video_path /path/to/your/video.mp4 --output output.mp4
```

### Özel Model Eğitimi

Özel bir veri seti üzerinde model eğitmek için:

```bash
python src/models/train_model.py --data your_dataset.csv --epochs 30 --output custom_model.pt
```

## İnsan Tespiti ve Takibi

Sistem, havuz ortamında kişileri tespit etmek ve izlemek için gelişmiş bir takip sistemi kullanmaktadır. Bu sistem aşağıdaki özelliklere sahiptir:

1. **YOLOv8 Tabanlı İnsan Tespiti**: Videodaki insanları tespit etmek için yüksek performanslı YOLOv8 modeli kullanılır.
2. **Havuz İçi/Dışı Ayrımı**: Sistem, tanımlanan havuz alanına göre kişilerin havuzun içinde mi yoksa dışında mı olduğunu belirler.
3. **Gelişmiş Kişi Takibi**: Her bireye bir ID atanır ve bu ID'ler sürekli izlenir, böylece farklı zamanlarda aynı kişinin hareketleri takip edilebilir.
4. **Hareket Yörüngeleri**: Kişilerin havuz içindeki hareket yönleri ve yörüngeleri analiz edilir.

### Havuz Alanı Tanımlama

Havuz alanı iki şekilde tanımlanabilir:
- **Otomatik Tespit**: `--auto_pool` parametresi ile video görüntüsünden havuz alanı otomatik olarak tespit edilmeye çalışılır.
- **Manuel Tanımlama**: `--pool_area "x1,y1,x2,y2"` parametresi ile havuz alanı koordinatları manuel olarak belirtilebilir.

### Performans Optimizasyonu

Sistem, daha hızlı işlem yapabilmek için çeşitli optimizasyon teknikleri içerir:
- **Çözünürlük Ölçekleme**: `--resize_factor` parametresi ile görüntüler işleme sırasında küçültülebilir (örneğin, 0.5=yarı boyut).
- **İşleme Hızı ve Doğruluk Dengesi**: Daha küçük ve hızlı modeller (yolov8n) veya daha doğru ancak yavaş modeller (yolov8x) arasında seçim yapılabilir.

### Kullanım Örnekleri

#### Temel Kullanım
```bash
python src/main.py --video_path /path/to/video.mp4 --output output.mp4
```

#### Manuel Havuz Alanı ile Kullanım
```bash
python src/main.py --video_path /path/to/video.mp4 --output output.mp4 --pool_area "100,100,540,320"
```

#### Otomatik Havuz Tespiti ile Kullanım
```bash
python src/main.py --video_path /path/to/video.mp4 --output output.mp4 --auto_pool
```

#### Performans İyileştirmeli Kullanım
```bash
python src/main.py --video_path /path/to/video.mp4 --output output.mp4 --pool_area "100,100,540,320" --resize_factor 0.5
```

## Modüller

### Veri Toplama
Video kaynakları ve görüntü yakalama işlemleri.

### İnsan Tespiti
YOLOv8 tabanlı nesne tanıma ve takip sistemi.

### Hareket Analizi
Yüzme hareketlerinin paternlerini analiz eden algoritma.

### Boğulma Tespiti
Anormal hareket paternlerini algılayan makine öğrenimi modeli.

### Görselleştirme
Tespit sonuçlarını ve uyarıları gösteren arayüz.

## Lisans

Bu proje [MIT lisansı](LICENSE) altında lisanslanmıştır. 