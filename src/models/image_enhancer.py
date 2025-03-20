import cv2
import numpy as np


class ImageEnhancer:
    """
    Görüntü iyileştirme teknikleri sınıfı.
    Özellikle havuz içi ve su altı görüntülerinin iyileştirilmesi için özel filtreler içerir.
    """
    
    def __init__(self):
        """Görüntü iyileştirici sınıfını başlatır."""
        pass
    
    def enhance_pool_image(self, image, method="combined", mask=None, **kwargs):
        """
        Havuz içi görüntü iyileştirme.
        
        Args:
            image: İşlenecek görüntü
            method: Kullanılacak yöntem ("clahe", "gamma", "unsharp", "dehaze", "combined")
            mask: Havuz maskesi (None ise tam görüntü işlenir)
            **kwargs: Yönteme özel parametreler
            
        Returns:
            İyileştirilmiş görüntü
        """
        if image is None:
            return None
            
        # Maskeyi hazırla
        if mask is not None:
            # Maskeyi 3 kanala genişlet
            if len(mask.shape) == 2:
                mask_3ch = cv2.merge([mask, mask, mask])
            else:
                mask_3ch = mask
                
            # Maskeli bölgeyi al
            pool_area = cv2.bitwise_and(image, image, mask=mask)
        else:
            pool_area = image.copy()
            mask_3ch = np.ones_like(image)
            
        # Seçilen yönteme göre iyileştirme uygula
        if method == "clahe":
            enhanced = self.apply_clahe(pool_area, **kwargs)
        elif method == "gamma":
            enhanced = self.apply_gamma_correction(pool_area, **kwargs)
        elif method == "unsharp":
            enhanced = self.apply_unsharp_masking(pool_area, **kwargs)
        elif method == "dehaze":
            enhanced = self.apply_dehazing(pool_area, **kwargs)
        elif method == "combined":
            # Birden fazla yöntemi birleştir
            enhanced = pool_area.copy()
            enhanced = self.apply_clahe(enhanced, clip_limit=kwargs.get("clip_limit", 2.0))
            enhanced = self.apply_gamma_correction(enhanced, gamma=kwargs.get("gamma", 1.2))
            enhanced = self.apply_unsharp_masking(enhanced, amount=kwargs.get("amount", 1.5))
        else:
            # Varsayılan olarak basit kontrast ve parlaklık ayarı
            contrast = kwargs.get("contrast", 1.3)
            brightness = kwargs.get("brightness", 10)
            enhanced = cv2.convertScaleAbs(pool_area, alpha=contrast, beta=brightness)
        
        # Sonucu orijinal görüntüyle birleştir
        if mask is not None:
            # Maskeli bölgeyi iyileştirilmiş görüntüyle değiştir
            result = image.copy()
            enhanced_region = cv2.bitwise_and(enhanced, mask_3ch)
            inverted_mask = cv2.bitwise_not(mask)
            inverted_mask_3ch = cv2.merge([inverted_mask, inverted_mask, inverted_mask])
            original_region = cv2.bitwise_and(image, inverted_mask_3ch)
            result = cv2.add(enhanced_region, original_region)
            return result
        else:
            return enhanced
    
    def apply_clahe(self, image, clip_limit=2.0, grid_size=(8, 8)):
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular.
        Kontrast sınırlı adaptif histogram eşitleme, lokal kontrast iyileştirmesi sağlar.
        
        Args:
            image: Giriş görüntüsü
            clip_limit: Histogram kırpma sınırı (1.0-4.0 arası)
            grid_size: Görüntüyü bölen ızgara boyutu
        
        Returns:
            İyileştirilmiş görüntü
        """
        # BGR görüntüyü LAB renk uzayına dönüştür
        # LAB uzayında L kanalı parlaklık, A ve B kanalları renk bilgisini içerir
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L kanalını ayır
        l, a, b = cv2.split(lab)
        
        # CLAHE oluştur ve uygula
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        cl = clahe.apply(l)
        
        # Kanalları birleştir
        limg = cv2.merge((cl, a, b))
        
        # LAB uzayından BGR'a geri dönüştür
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
        
    def apply_gamma_correction(self, image, gamma=1.2):
        """
        Gamma düzeltmesi uygular.
        Görüntünün karanlık bölgelerini aydınlatmak veya aydınlık bölgeleri karartmak için kullanılır.
        
        Args:
            image: Giriş görüntüsü
            gamma: Gamma değeri (1'den küçükse karartır, 1'den büyükse aydınlatır)
            
        Returns:
            İyileştirilmiş görüntü
        """
        # Görüntüyü normalize et (0-1 aralığına çevir)
        normalized = image / 255.0
        
        # Gamma düzeltmesi uygula
        corrected = np.power(normalized, 1.0/gamma)
        
        # Tekrar 0-255 aralığına dönüştür ve uint8 tipine çevir
        enhanced = np.uint8(corrected * 255)
        
        return enhanced
        
    def apply_unsharp_masking(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
        """
        Unsharp masking (keskinleştirme) filtresini uygular.
        Görüntüdeki kenarları ve detayları vurgular.
        
        Args:
            image: Giriş görüntüsü
            kernel_size: Bulanıklaştırma çekirdeği boyutu
            sigma: Gaussian bulanıklaştırma standart sapması
            amount: Keskinleştirme miktarı (1.0-2.0 arası)
            threshold: Uygulama eşiği (0-255 arası)
            
        Returns:
            Keskinleştirilmiş görüntü
        """
        # Gaussian bulanıklaştırma uygula
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # Unsharp mask oluştur (Orijinal - Bulanık)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        
        # Değerleri sınırla
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        
        # Eşik değerine göre filtreleme (isteğe bağlı)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            sharpened[low_contrast_mask] = image[low_contrast_mask]
        
        return sharpened
        
    def apply_dehazing(self, image, omega=0.95, t0=0.1, radius=15):
        """
        Görüntüdeki sis/bulanıklık etkisini azaltır (Dark Channel Prior yöntemi).
        Su altı görüntülerinde bulanıklığı azaltmak için faydalıdır.
        
        Args:
            image: Giriş görüntüsü
            omega: Sis kaldırma ağırlığı (0-1 arası)
            t0: Minimum iletim değeri
            radius: Karanlık kanal hesabı için yarıçap
            
        Returns:
            Sis giderilmiş görüntü
        """
        # Görüntü boyutları
        h, w, _ = image.shape
        
        # Minimum filter ile karanlık kanalı hesapla
        dark_channel = np.min(image, axis=2)
        
        # Karanlık kanalı çekirdek boyutu ile erode et
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
        dark_channel = cv2.erode(dark_channel, kernel)
        
        # Atmosferik ışığı tahmin et (en parlak pikseller)
        num_pixels = int(0.001 * h * w)
        flat_img = image.reshape(h*w, 3)
        flat_dark = dark_channel.ravel()
        
        # En yüksek yoğunluğa sahip piksellerin indeksleri
        indices = np.argsort(flat_dark)[-num_pixels:]
        atmospheric = np.max(flat_img.take(indices, axis=0), axis=0)
        
        # İletim haritasını hesapla
        transmission = 1 - omega * dark_channel / np.max(atmospheric)
        
        # İletim değerlerini t0 ile sınırla
        transmission = np.maximum(transmission, t0)
        
        # Sisden arındırılmış görüntüyü oluştur
        dehazed = np.zeros_like(image, dtype=np.float32)
        
        for i in range(3):
            dehazed[:,:,i] = (image[:,:,i] - atmospheric[i]) / transmission + atmospheric[i]
        
        # Değerleri sınırla ve uint8'e dönüştür
        dehazed = np.clip(dehazed, 0, 255).astype(np.uint8)
        
        return dehazed
    
    def detect_underwater_regions(self, image, blue_thresh=1.1, sat_thresh=0.3):
        """
        Görüntüdeki olası su içi/altı bölgeleri tespit eder.
        
        Args:
            image: Giriş görüntüsü
            blue_thresh: Mavi kanal ağırlık eşiği
            sat_thresh: Doygunluk eşiği
            
        Returns:
            Su içi/altı bölgelerinin maskesi
        """
        # BGR > HSV dönüşümü
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # BGR kanallarını ayır
        b, g, r = cv2.split(image)
        
        # Mavi kanalın ağırlığını hesapla
        blue_dominance = np.zeros_like(b, dtype=np.float32)
        # Sıfıra bölünmeyi önle
        safe_r = np.maximum(r, 1)
        safe_g = np.maximum(g, 1)
        blue_dominance = b / ((safe_r + safe_g) / 2)
        
        # Su içi/altı bölgelerini tespit et
        # 1. Mavi kanal baskın
        # 2. Doygunluk yüksek değil (su içindeki nesneler daha az doygun görünür)
        underwater_mask = np.logical_and(
            blue_dominance > blue_thresh,
            s < sat_thresh * 255
        ).astype(np.uint8) * 255
        
        # Morfolojik işlemler
        kernel = np.ones((5, 5), np.uint8)
        underwater_mask = cv2.morphologyEx(underwater_mask, cv2.MORPH_OPEN, kernel)
        underwater_mask = cv2.morphologyEx(underwater_mask, cv2.MORPH_CLOSE, kernel)
        
        return underwater_mask 