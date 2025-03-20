#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Görselleştirme Modülü
--------------------
Tespit sonuçlarını görselleştirmek için kullanılan sınıf.
"""

import cv2
import numpy as np
import time


class Visualizer:
    """Tespit sonuçlarını görselleştiren sınıf."""
    
    def __init__(self, show_tracking=True, show_stats=True, show_pool_area=True):
        """
        Visualizer sınıfını başlatır.
        
        Args:
            show_tracking (bool): Takip çizgilerinin gösterilip gösterilmeyeceği
            show_stats (bool): İstatistiklerin gösterilip gösterilmeyeceği
            show_pool_area (bool): Havuz alanının gösterilip gösterilmeyeceği
        """
        self.show_tracking = show_tracking
        self.show_stats = show_stats
        self.show_pool_area = show_pool_area
        
        # Renkler
        self.alert_color = (0, 0, 255)  # Kırmızı (BGR)
        self.detection_color = (0, 255, 0)  # Yeşil (BGR)
        self.pool_person_color = (255, 128, 0)  # Mavi-Turuncu (BGR)
        self.track_color = (255, 255, 0)  # Açık mavi (BGR)
        self.text_color = (255, 255, 255)  # Beyaz (BGR)
        self.pool_color = (255, 153, 51)  # Mavi-turuncu (BGR)
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        
        # Son uyarı zamanı
        self.last_alert_time = 0
        self.alert_duration = 2  # saniye
        
        # Havuz alanı
        self.pool_area = None  # [x1, y1, x2, y2] formatında havuz alanı
        self.pool_polygon = None  # [(x1,y1), (x2,y2), ...] formatında nokta listesi
    
    def set_pool_area(self, pool_area):
        """
        Havuz alanını ayarlar.
        
        Args:
            pool_area (list): [x1, y1, x2, y2] formatında havuz alanı koordinatları
        """
        self.pool_area = pool_area
    
    def set_pool_polygon(self, pool_polygon):
        """
        Çokgen havuz alanını ayarlar.
        
        Args:
            pool_polygon (list): [(x1,y1), (x2,y2), ...] formatında nokta listesi
        """
        self.pool_polygon = pool_polygon
    
    def draw(self, frame, detections, tracking_info=None, alerts=None):
        """
        Tespit sonuçlarını ve uyarıları kare üzerine çizer.
        
        Args:
            frame (ndarray): Video karesi
            detections (list): [(id, bbox, confidence), ...] formatında tespit listesi
            tracking_info (dict, optional): Nesne ID'lerine göre takip bilgileri
            alerts (list, optional): [(id, bbox, score), ...] formatında uyarı listesi
            
        Returns:
            ndarray: Çizimler eklenmiş video karesi
        """
        # Kareyi kopyala (orijinali değiştirmemek için)
        output = frame.copy()
        
        # Havuz alanını çiz
        if self.show_pool_area:
            if self.pool_polygon:
                self._draw_pool_polygon(output, self.pool_polygon)
            elif self.pool_area:
                self._draw_pool_area(output, self.pool_area)
        
        # Kaç kişi havuz içinde ve dışında
        pool_person_count = 0
        total_person_count = len(detections)
        
        # Tüm tespitleri çiz
        for detection in detections:
            person_id, bbox, confidence = detection
            
            # Kişi havuz içinde mi?
            is_in_pool = False
            if tracking_info is not None and person_id in tracking_info:
                is_in_pool = tracking_info[person_id].get('is_in_pool', False)
                if is_in_pool:
                    pool_person_count += 1
            
            # Havuz içindeki kişiler için farklı renk kullan
            self._draw_detection(output, person_id, bbox, confidence, False, is_in_pool)
            
            # Hareket geçmişini çiz
            if self.show_tracking and tracking_info is not None and person_id in tracking_info:
                center_points = tracking_info[person_id].get('center_points', [])
                if center_points:
                    self._draw_tracking_path(output, list(center_points), is_in_pool)
        
        # Uyarıları çiz
        if alerts:
            for alert in alerts:
                person_id, bbox, score = alert
                self._draw_detection(output, person_id, bbox, score, True)
                
                # Ekranın üstünde uyarı yazısı göster
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_duration:
                    self.last_alert_time = current_time
                
                if current_time - self.last_alert_time < self.alert_duration:
                    cv2.putText(output, "! BOĞULMA TESPİT EDİLDİ !", 
                                (int(output.shape[1]/2) - 150, 70), 
                                self.font, 1.2, (0, 0, 255), 3)
        
        # İstatistikleri göster
        if self.show_stats:
            self._draw_stats(output, total_person_count, pool_person_count, len(alerts) if alerts else 0)
        
        return output
    
    def _draw_detection(self, frame, person_id, bbox, score, is_alert, is_in_pool=False):
        """
        Bir tespiti kare üzerine çizer.
        
        Args:
            frame (ndarray): Video karesi
            person_id (int): Kişi ID'si
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            score (float): Tespit veya uyarı skoru
            is_alert (bool): Uyarı tespiti mi
            is_in_pool (bool): Kişi havuz içinde mi
        """
        x1, y1, x2, y2 = bbox
        
        # Tespit türüne göre renk belirle
        if is_alert:
            color = self.alert_color
        elif is_in_pool:
            color = self.pool_person_color
        else:
            color = self.detection_color
        
        # Sınırlayıcı kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # Kişi ID'si ve skoru etikette göster
        if is_in_pool:
            label = f"ID: {person_id} ({score:.2f}) [HAVUZ]"
        else:
            label = f"ID: {person_id} ({score:.2f})"
            
        if is_alert:
            label = f"! BOĞULMA ! ID: {person_id} ({score:.2f})"
        
        # Etiket için arka plan dikdörtgeni
        label_size, baseline = cv2.getTextSize(label, self.font, self.font_scale, 1)
        y1 = max(y1, label_size[1])
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Etiketi çiz
        cv2.putText(frame, label, (x1, y1 - 7), self.font, self.font_scale, self.text_color, 1)
        
        # Merkezi işaretle
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    def _draw_tracking_path(self, frame, center_points, is_in_pool=False):
        """
        Hareket yolunu çizer.
        
        Args:
            frame (ndarray): Video karesi
            center_points (list): (x, y) formatında merkez nokta listesi
            is_in_pool (bool): Kişi havuz içinde mi
        """
        if len(center_points) < 2:
            return
        
        # Havuzdaki kişilerin izini daha kalın çiz
        thickness = 3 if is_in_pool else 2
        
        # Renk havuzdaki kişiler için daha parlak
        color = self.pool_person_color if is_in_pool else self.track_color
        
        # Son N nokta için yolun rengini koyulaştır
        max_points = min(len(center_points), 20)  # Son 20 noktayı kullan
        
        # Noktaları numpy dizisine dönüştür
        points = np.array(center_points[-max_points:], np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Yol çizgisini çiz
        cv2.polylines(frame, [points], False, color, thickness)
    
    def _draw_stats(self, frame, num_people, num_in_pool, num_alerts):
        """
        İstatistikleri ekranın altına çizer.
        
        Args:
            frame (ndarray): Video karesi
            num_people (int): Tespit edilen toplam kişi sayısı
            num_in_pool (int): Havuz içindeki kişi sayısı
            num_alerts (int): Aktif uyarı sayısı
        """
        h, w = frame.shape[:2]
        
        # Arka plan dikdörtgeni
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        
        # İstatistikleri çiz
        stats_text = f"Toplam: {num_people} kişi | Havuz içi: {num_in_pool} kişi | Uyarı: {num_alerts}"
        cv2.putText(frame, stats_text, (10, h - 15), self.font, 0.6, self.text_color, 1)
        
        # Tarih ve saati çiz
        time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        time_size = cv2.getTextSize(time_text, self.font, 0.6, 1)[0]
        cv2.putText(frame, time_text, (w - time_size[0] - 10, h - 15), self.font, 0.6, self.text_color, 1)
    
    def _draw_pool_area(self, frame, pool_area):
        """
        Havuz alanını dikdörtgen olarak çizer.
        
        Args:
            frame (ndarray): Video karesi
            pool_area (list): [x1, y1, x2, y2] formatında havuz alanı koordinatları
        """
        x1, y1, x2, y2 = pool_area
        
        # Yarı saydam bir havuz alanı dikdörtgeni çiz
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.pool_color, -1)  # Mavi-turuncu dolgu
        
        # Havuz alanı etrafına çerçeve çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.pool_color, 2)  # Mavi-turuncu çerçeve
        
        # Yarı saydamlık için iki resmi karıştır
        alpha = 0.2  # Saydamlık derecesi
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # "HAVUZ ALANI" etiketi
        label = "HAVUZ ALANI"
        label_size, baseline = cv2.getTextSize(label, self.font, 0.7, 2)
        cv2.putText(frame, label, (x1 + 10, y1 + 25), self.font, 0.7, (255, 255, 255), 2)
    
    def _draw_pool_polygon(self, frame, pool_polygon):
        """
        Havuz alanını çokgen olarak çizer.
        
        Args:
            frame (ndarray): Video karesi
            pool_polygon (list): [(x1,y1), (x2,y2), ...] formatında nokta listesi
        """
        if not pool_polygon or len(pool_polygon) < 3:
            return
        
        # Numpy dizisine dönüştür
        points = np.array(pool_polygon, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Yarı saydam bir havuz alanı çokgeni çiz
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], self.pool_color)  # Mavi-turuncu dolgu
        
        # Havuz alanı etrafına çerçeve çiz
        cv2.polylines(frame, [points], True, self.pool_color, 2)  # Mavi-turuncu çerçeve
        
        # Yarı saydamlık için iki resmi karıştır
        alpha = 0.2  # Saydamlık derecesi
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # "HAVUZ ALANI" etiketi
        # Çokgenin merkez noktasını hesapla
        centroid_x = int(np.mean([p[0] for p in pool_polygon]))
        centroid_y = int(np.mean([p[1] for p in pool_polygon]))
        
        label = "HAVUZ ALANI"
        label_size, baseline = cv2.getTextSize(label, self.font, 0.7, 2)
        cv2.putText(frame, label, (centroid_x - label_size[0]//2, centroid_y), 
                    self.font, 0.7, (255, 255, 255), 2)
        
        # Çokgen noktalarını göster
        for i, (x, y) in enumerate(pool_polygon):
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Sarı nokta
            cv2.putText(frame, str(i+1), (x+5, y+5), self.font, 0.5, (0, 255, 255), 1)
    
    def reset_tracking(self):
        """
        Takip sistemini sıfırlar.
        """
        self.track_history = {} 