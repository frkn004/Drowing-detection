#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boğulma Tespiti Modülü
----------------------
Hareket analizi verilerini kullanarak boğulma vakalarını tespit eden sınıf.
"""

import numpy as np
import time
from collections import deque


class DrowningDetector:
    """Boğulma belirtilerini tespit eden sınıf."""
    
    def __init__(self, config=None):
        """
        DrowningDetector sınıfını başlatır.
        
        Args:
            config (dict): Konfigürasyon parametreleri
        """
        self.config = config or {}
        
        # Varsayılan parametreler
        self.stillness_threshold = self.config.get('stillness_threshold', 15)  # Hareketsizlik eşiği (kare sayısı)
        self.erratic_threshold = self.config.get('erratic_threshold', 8)       # Düzensiz hareket eşiği
        self.alert_duration = self.config.get('alert_duration', 3)             # Uyarı süresi (saniye)
        
        # Aktif uyarılar
        self.active_alerts = {}
        
        # Boğulma modelinin durumu (basit kural tabanlı veya ML tabanlı olabilir)
        self.model_type = self.config.get('model_type', 'rule_based')
        
        # Eğer özel model varsa yükle
        if self.model_type == 'ml':
            try:
                import torch
                model_path = self.config.get('model_path', 'models/drowning_detector.pt')
                self.ml_model = torch.load(model_path)
                self.ml_model.eval()
                print(f"Boğulma tespit modeli yüklendi: {model_path}")
            except Exception as e:
                print(f"Model yüklenirken hata oluştu: {e}")
                print("Kural tabanlı modele geçiliyor")
                self.model_type = 'rule_based'
    
    def detect(self, movement_data, person_id, is_in_pool=False):
        """
        Boğulma belirtilerini tespit eder.
        
        Args:
            movement_data (dict/list): Hareket analiz verileri
            person_id (int): Kişi ID'si
            is_in_pool (bool): Kişinin havuz içinde olup olmadığı
            
        Returns:
            tuple: (boğulma_skoru, boğulma_durumu) çifti
        """
        # Eğer kişi havuz içinde değilse, boğulma riski yoktur
        if not is_in_pool:
            return 0.0, False
            
        # Geçersiz veya yetersiz hareket verisi varsa
        if not movement_data or (isinstance(movement_data, list) and len(movement_data) < 5):
            return 0.0, False
            
        # Mevcut zaman
        current_time = time.time()
        
        # Daha önce uyarı verildiyse ve süresi dolmadıysa, aynı sonucu döndür
        if person_id in self.active_alerts:
            alert_time, alert_score = self.active_alerts[person_id]
            if current_time - alert_time < self.alert_duration:
                return alert_score, True
        
        # Veri formatına göre farklı değerlendirme metodları kullan
        if isinstance(movement_data, list):
            # Liste formatı (merkez noktalar listesi)
            return self._process_movement_data_list(movement_data, person_id)
        else:
            # Sözlük formatı (hareket analiz verileri)
            return self._rule_based_detection(movement_data, person_id)
            
    def _process_movement_data_list(self, movement_data, person_id):
        """
        Liste formatındaki hareket verilerini işler ve boğulma tespiti yapar.
        
        Args:
            movement_data (list): [(x1,y1), (x2,y2), ...] formatında merkez noktalar
            person_id (int): Kişi ID'si
            
        Returns:
            tuple: (boğulma_skoru, boğulma_durumu) çifti
        """
        # En az 5 nokta olmalı
        if len(movement_data) < 5:
            return 0.0, False
            
        # Son 10 noktayı al (eğer varsa)
        points = movement_data[-10:] if len(movement_data) > 10 else movement_data
        
        # Merkez noktaların hareketi analiz et
        displacements = []
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            # Öklid mesafesi
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            displacements.append(dist)
            
        # Hareket istatistikleri
        avg_displacement = np.mean(displacements) if displacements else 0
        
        # Hareketsizlik kontrolü (çok küçük hareketler)
        stillness_count = sum(1 for d in displacements if d < self.stillness_threshold)
        
        # Düzensiz hareket kontrolü (çok büyük değişimler)
        erratic_count = 0
        for i in range(1, len(displacements)):
            if abs(displacements[i] - displacements[i-1]) > self.erratic_threshold:
                erratic_count += 1
                
        # Toplam puan hesapla
        # Hareketsizlik puanı (0-60) - daha uzun hareketsizlik daha yüksek puan
        stillness_score = min(60, (stillness_count / len(displacements)) * 100) if displacements else 0
        
        # Düzensiz hareket puanı (0-40) - daha fazla düzensizlik daha yüksek puan
        erratic_score = min(40, (erratic_count / max(1, len(displacements)-1)) * 100)
        
        # Toplam boğulma skoru (0-100)
        drowning_score = stillness_score + erratic_score
        
        # Boğulma tespit eşiği (70)
        is_drowning = drowning_score > 70
        
        # Eğer boğulma tespit edildiyse, aktif uyarıları güncelle
        if is_drowning:
            self.active_alerts[person_id] = (time.time(), drowning_score / 100.0)
            
        return drowning_score / 100.0, is_drowning
    
    def _rule_based_detection(self, movement_data, person_id):
        """
        Kural tabanlı boğulma tespiti.
        
        Args:
            movement_data (dict): Hareket analizi verileri
            person_id (int): Kişi ID'si
            
        Returns:
            tuple: (boğulma_skoru, boğulma_durumu) çifti
        """
        try:
            # Hareket verilerini kontrol et
            if not movement_data:
                return 0.0, False
                
            # Hareketsizlik sayacı
            stillness_count = 0
            if isinstance(movement_data, dict) and 'stillness_count' in movement_data:
                stillness_count = movement_data.get('stillness_count', 0)
            
            # Düzensiz hareket sayacı
            erratic_count = 0
            if isinstance(movement_data, dict) and 'erratic_count' in movement_data:
                erratic_count = movement_data.get('erratic_count', 0)
            
            # Toplam puan hesapla
            # Hareketsizlik puanı (0-60) - daha uzun hareketsizlik daha yüksek puan
            stillness_score = min(60, stillness_count * 4)
            
            # Düzensiz hareket puanı (0-40) - daha fazla düzensizlik daha yüksek puan
            erratic_score = min(40, erratic_count * 8)
            
            # Toplam boğulma skoru (0-100)
            drowning_score = stillness_score + erratic_score
            
            # Boğulma tespit eşiği (70)
            is_drowning = drowning_score > 70
            
            # Eğer boğulma tespit edildiyse, aktif uyarıları güncelle
            if is_drowning:
                self.active_alerts[person_id] = (time.time(), drowning_score / 100.0)
                
            return drowning_score / 100.0, is_drowning
            
        except Exception as e:
            print(f"Boğulma tespiti hatası: {str(e)}")
            return 0.0, False
    
    def _ml_based_detection(self, movement_data, person_id):
        """
        Makine öğrenmesi tabanlı boğulma tespiti.
        
        Args:
            movement_data (dict): Hareket analizi verileri
            person_id (int): Kişi ID'si
            
        Returns:
            tuple: (drowning_score, is_drowning) - boğulma skoru ve durum booleani
        """
        try:
            import torch
            
            # Hareket verilerinden özellik vektörü oluştur
            features = [
                movement_data.get('speed', 0),
                movement_data.get('avg_speed', 0),
                movement_data.get('acceleration', 0),
                movement_data.get('movement_variance', 0),
                movement_data.get('stillness_count', 0),
                movement_data.get('erratic_count', 0)
            ]
            
            # Özellik vektörünü tensor'a dönüştür
            features_tensor = torch.FloatTensor(features).unsqueeze(0)  # batch boyutu ekle
            
            # Model tahmini yap
            with torch.no_grad():
                drowning_score = self.ml_model(features_tensor).item()
            
            # Skoru uyarı geçmişine ekle
            self.active_alerts[person_id]['history'].append(drowning_score)
            
            # Son N karede ortalama skoru hesapla (kararlılık için)
            avg_score = np.mean(self.active_alerts[person_id]['history']) if self.active_alerts[person_id]['history'] else 0
            
            # Uyarı durumunu güncelle
            current_time = time.time()
            
            if avg_score > 0.6 and current_time - self.active_alerts[person_id]['last_alert_time'] > self.alert_duration:
                self.active_alerts[person_id]['is_drowning'] = True
                self.active_alerts[person_id]['last_alert_time'] = current_time
                self.active_alerts[person_id]['alert_count'] += 1
            elif avg_score < 0.3:
                self.active_alerts[person_id]['is_drowning'] = False
            
            return avg_score, self.active_alerts[person_id]['is_drowning']
            
        except Exception as e:
            print(f"ML tabanlı tespitte hata: {e}")
            # Hata durumunda kural tabanlı tespite geri dön
            return self._rule_based_detection(movement_data, person_id)
    
    def get_all_alerts(self):
        """
        Tüm uyarıları döndürür.
        
        Returns:
            dict: Kişi ID'lerini anahtar olarak içeren uyarı bilgileri sözlüğü
        """
        return self.active_alerts
    
    def reset_alerts(self, person_id=None):
        """
        Uyarıları sıfırlar.
        
        Args:
            person_id (int, optional): Sıfırlanacak kişi ID'si. None ise tüm uyarılar sıfırlanır.
        """
        if person_id is None:
            self.active_alerts = {}
        elif person_id in self.active_alerts:
            del self.active_alerts[person_id] 