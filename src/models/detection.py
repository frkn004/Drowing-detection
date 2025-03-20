#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
İnsan Tespiti Modülü
-------------------
Havuzdaki insanları tespit etmek için YOLOv8 gibi
hazır nesne tespit modellerini kullanan sınıf.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import time
import os


class KalmanFilter:
    """
    Basitleştirilmiş Kalman Filtresi sınıfı.
    Nesne takibinde daha düzgün hareket için kullanılır.
    """
    
    def __init__(self, bbox):
        """
        Kalman Filtresini başlatır.
        
        Args:
            bbox (list): Başlangıç bounding box [x1, y1, x2, y2]
        """
        # Durum vektörü: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.kalman = cv2.KalmanFilter(8, 4)
        
        # Geçiş matrisi: Pozisyon ve hız modelini tanımlar
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1 = x1 + vx1
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1 = y1 + vy1
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2 = x2 + vx2
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2 = y2 + vy2
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx1
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy1
            [0, 0, 0, 0, 0, 0, 1, 0],  # vx2
            [0, 0, 0, 0, 0, 0, 0, 1]   # vy2
        ], np.float32)
        
        # Ölçüm matrisi: Sadece pozisyonlar ölçülür
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        # İşlem gürültüsü kovaryansı
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Ölçüm gürültüsü kovaryansı
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Başlangıç durumu
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32) * 1
        
        # Durum vektörünü başlat
        self.kalman.statePost = np.array([
            [bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]
        ], dtype=np.float32)
    
    def predict(self):
        """
        Bir sonraki durumu tahmin et.
        
        Returns:
            list: Tahmin edilen bounding box [x1, y1, x2, y2]
        """
        # Kalman filtresi tahmini
        state = self.kalman.predict()
        
        # Tahmin edilen bounding box değerlerini döndür
        return [
            max(0, int(state[0][0])),
            max(0, int(state[1][0])),
            max(0, int(state[2][0])),
            max(0, int(state[3][0]))
        ]
    
    def update(self, bbox):
        """
        Ölçüm ile filtreyi güncelle.
        
        Args:
            bbox (list): Ölçülen bounding box [x1, y1, x2, y2]
        """
        # Ölçüm vektörü oluştur
        measurement = np.array([
            [bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]
        ], dtype=np.float32)
        
        # Kalman filtresi düzeltmesi
        self.kalman.correct(measurement)


class PersonDetector:
    """YOLO ile insan tespiti ve takibi sınıfı."""
    
    def __init__(self, model=None, model_name='yolov8m', device=None, threshold=0.3, 
                 max_disappeared=30, iou_threshold=0.3, area_threshold=100, use_kalman=False):
        """
        Tespit ve takip için sınıf başlatıcı.
        
        Args:
            model: Önceden yüklenmiş bir YOLO modeli (None ise model_name kullanılır)
            model_name (str): Kullanılacak model ismi. 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            device (str): CPU veya CUDA cihazı
            threshold (float): Tespit eşik değeri
            max_disappeared (int): Bir nesnenin kaç kare kaybolabileceği
            iou_threshold (float): IoU eşik değeri
            area_threshold (int): Minimum tespit alanı (piksel cinsinden)
            use_kalman (bool): Kalman filtresi kullanıp kullanmama
        """
        # Cihaz kontrolü
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # YOLO modeli
        if model is not None:
            self.model = model
        else:
            # Model yolunu oluştur
            model_path = f"{model_name}.pt"
            if not os.path.exists(model_path) and model_name.startswith('yolov8'):
                # Model yüklenmemişse indirmeyi dene
                print(f"Model dosyası bulunamadı: {model_path}, indiriliyor...")
                from ultralytics import YOLO
                self.model = YOLO(model_name)
            else:
                # Yerel model dosyasını yükle
                print(f"Model yükleniyor: {model_path}")
                from ultralytics import YOLO
                self.model = YOLO(model_path)
        
        # Yapılandırma
        self.threshold = threshold
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.area_threshold = area_threshold
        self.use_kalman = use_kalman
        self.track_buffer = 30  # Takip geçmişi tampon boyutu
        
        # Takip veri yapıları
        self.next_object_id = 0
        self.tracked_objects = {}  # {ID: {bbox, centroid, disappeared, ...}}
        
        # Kalman filtresi veri yapıları
        self.kalman_filters = {}  # {ID: KalmanFilter}
        
        # Hareket analizi veri yapıları
        self.movement_data = {}  # {ID: [center_points]}
        
        print(f"PersonDetector başlatıldı: model={model_name}, threshold={threshold}, "
              f"iou_threshold={iou_threshold}, area_threshold={area_threshold}, "
              f"use_kalman={use_kalman}")
        
        # Kişi takibi için ID yönetimi
        self.next_id = 0
        self.tracked_objects = {}  # id -> {bbox, last_seen, disappeared, center_points, conf_history}
        
        # Havuz içi ve kenarı alanları (gerekirse ayarlanabilir)
        self.pool_area = None  # [x1, y1, x2, y2] formatında havuz alanı
        
        # Kalman filtresi takibi için
        if self.use_kalman:
            self.kalman_filters = {}
        
    def set_pool_area(self, pool_area):
        """
        Havuz alanını ayarlar (havuz içi ve kenarı bölgeleri)
        
        Args:
            pool_area (list): [x1, y1, x2, y2] formatında havuz alanı koordinatları
        """
        self.pool_area = pool_area
        print(f"Havuz alanı ayarlandı: {pool_area}")
        
    def detect(self, frame):
        """
        Görüntü üzerinde insan tespiti yapar.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            
        Returns:
            list: [(id, bbox, confidence), ...] formatında tespit listesi
        """
        # Boş kare kontrolü
        if frame is None or frame.size == 0:
            return []
        
        # YOLO ile tespit
        results = self.model(
            frame, 
            conf=self.threshold,
            verbose=False,
            classes=[0],  # Sadece insan (0) sınıfı için tespit
            iou=self.iou_threshold,
            max_det=50,   # Maksimum tespit sayısı
            agnostic_nms=True  # Sınıftan bağımsız NMS (Non-Maximum Suppression)
        )
        
        # Tespitleri al
        detections = []
        persons = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Sadece insan sınıfı
                if box.cls == 0:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Küçük nesneleri filtrele
                    area = (x2 - x1) * (y2 - y1)
                    if area < self.area_threshold:
                        continue
                    
                    # Sınırlayıcı kutuyu kare sınırları içinde tut
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Confidence
                    conf = float(box.conf.cpu().numpy())
                    
                    # Tespit listesine ekle
                    persons.append((None, [x1, y1, x2, y2], conf))
        
        # Yeni tespitlerle takip sistemini güncelle
        self._update_all_tracked_objects(persons)
        
        # Güncel takip listesini döndür
        detections = []
        for person_id, obj_info in self.tracked_objects.items():
            if obj_info['disappeared'] == 0:  # Sadece mevcut kareler için
                detections.append((person_id, obj_info['bbox'], obj_info['confidence']))
        
        return detections

    def detect_in_pool(self, frame, pool_mask=None, pool_threshold=0.2):
        """
        Havuz içerisindeki insanları daha düşük threshold ile tespit eder.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            pool_mask (ndarray): Havuz alanı maskesi (None ise havuz maskesi kullanılmaz)
            pool_threshold (float): Havuz içi tespit eşik değeri
            
        Returns:
            list: [(id, bbox, confidence), ...] formatında tespit listesi
        """
        # Boş kare kontrolü
        if frame is None or frame.size == 0:
            return []
        
        # YOLO ile tespit (düşük eşik değeri)
        results = self.model(
            frame, 
            conf=pool_threshold,  # Düşük eşik değeri
            verbose=False,
            classes=[0],  # Sadece insan (0) sınıfı için tespit
            iou=self.iou_threshold,
            max_det=50,   # Maksimum tespit sayısı
            agnostic_nms=True  # Sınıftan bağımsız NMS
        )
        
        # Tespitleri al
        persons = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Sadece insan sınıfı
                if box.cls == 0:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Küçük nesneleri filtrele (havuz içinde daha küçük nesnelere izin ver)
                    area = (x2 - x1) * (y2 - y1)
                    min_area = self.area_threshold * 0.7  # Havuz içinde daha küçük alanlara izin ver
                    if area < min_area:
                        continue
                    
                    # Sınırlayıcı kutuyu kare sınırları içinde tut
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Merkez noktayı hesapla
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Havuz maskesi varsa, merkez noktanın havuz içinde olup olmadığını kontrol et
                    is_in_pool = True
                    if pool_mask is not None:
                        if 0 <= cy < pool_mask.shape[0] and 0 <= cx < pool_mask.shape[1]:
                            is_in_pool = pool_mask[cy, cx] > 0.5
                        else:
                            is_in_pool = False
                    
                    # Sadece havuz içindeki tespitleri ekle
                    if is_in_pool:
                        # Confidence
                        conf = float(box.conf.cpu().numpy())
                        
                        # Tespit listesine ekle
                        persons.append((None, [x1, y1, x2, y2], conf))
        
        # Yeni tespitlerle takip sistemini güncelle
        self._update_all_tracked_objects(persons)
        
        # Güncel takip listesini döndür
        detections = []
        for person_id, obj_info in self.tracked_objects.items():
            if obj_info['disappeared'] == 0:  # Sadece mevcut kareler için
                detections.append((person_id, obj_info['bbox'], obj_info['confidence']))
        
        return detections
    
    def _update_tracked_object(self, obj_id, bbox, confidence):
        """
        Takip edilen bir nesneyi günceller.
        
        Args:
            obj_id (int): Nesne kimliği
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            confidence (float): Tespit güven değeri
        """
        # Nesne bilgilerini güncelle
        self.tracked_objects[obj_id]['bbox'] = bbox
        self.tracked_objects[obj_id]['confidence'] = confidence
        self.tracked_objects[obj_id]['disappeared'] = 0
        
        # Merkez noktayı hesapla
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Merkez noktayı hareket listesine ekle
        if obj_id not in self.movement_data:
            self.movement_data[obj_id] = []
        self.movement_data[obj_id].append((cx, cy))
        
        # Son 30 noktayı tut
        if len(self.movement_data[obj_id]) > 30:
            self.movement_data[obj_id] = self.movement_data[obj_id][-30:]
        
        # Kalman filtresi varsa güncelle
        if self.use_kalman and obj_id in self.kalman_filters:
            kf = self.kalman_filters[obj_id]
            kf.update([x1, y1, x2, y2])
        
        # Havuzda mı değil mi güncelle
        self.tracked_objects[obj_id]['is_in_pool'] = self._is_in_pool_area(bbox)
        
        # Merkez noktayı geçmişe ekle
        self.tracked_objects[obj_id]['center_points'].append((cx, cy))
        # Güven değerini geçmişe ekle
        self.tracked_objects[obj_id]['conf_history'].append(confidence)
    
    def _register_new_object(self, bbox, confidence):
        """
        Yeni bir nesneyi takip edilenler listesine kaydeder.
        
        Args:
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            confidence (float): Tespit güven değeri
            
        Returns:
            int: Yeni oluşturulan nesne ID'si
        """
        # Yeni nesne ID'si ata
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        # Merkez noktayı hesapla
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Nesne bilgilerini kaydet
        self.tracked_objects[obj_id] = {
            'bbox': bbox,
            'confidence': confidence,
            'disappeared': 0,
            'is_in_pool': self._is_in_pool_area(bbox),
            'center_points': deque(maxlen=30),
            'conf_history': deque([confidence], maxlen=30)
        }
        
        # Merkez noktayı geçmişe ekle
        self.tracked_objects[obj_id]['center_points'].append((cx, cy))
        
        # Hareket bilgisini başlat
        self.movement_data[obj_id] = [(cx, cy)]
        
        # Kalman filtresi oluştur (eğer aktifse)
        if self.use_kalman:
            kf = KalmanFilter(bbox)
            self.kalman_filters[obj_id] = kf
        
        return obj_id
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        İki sınırlayıcı kutu arasındaki IoU (Intersection over Union) değerini hesaplar.
        
        Args:
            bbox1 (list): [x1, y1, x2, y2] formatında ilk sınırlayıcı kutu
            bbox2 (list): [x1, y1, x2, y2] formatında ikinci sınırlayıcı kutu
            
        Returns:
            float: IoU değeri
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Kesişim alanı
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Kesişim alanı yok
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Birleşim alanı
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # IoU hesapla
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _greedy_assignment(self, cost_matrix, threshold=0.3):
        """
        Açgözlü atama algoritması (Macar algoritmasının basit versiyonu)
        
        Args:
            cost_matrix (ndarray): Maliyet matrisi (IoU değerleri)
            threshold (float): Eşleştirme eşik değeri
            
        Returns:
            list: (row, col) formatında eşleşen indisler listesi
        """
        # IoU değerlerini -1 ile çarp (maksimize etmek istiyoruz)
        cost_matrix = -cost_matrix.copy()
        
        # Eşik değerinden düşük IoU değerlerini çok yüksek maliyet olarak ata
        cost_matrix[cost_matrix > -threshold] = float('inf')
        
        # Satır ve sütunların kopyası
        unassigned_rows = list(range(cost_matrix.shape[0]))
        unassigned_cols = list(range(cost_matrix.shape[1]))
        
        assignments = []
        
        # Tüm satırlar atanana veya tüm sütunlar atanana kadar devam et
        while unassigned_rows and unassigned_cols:
            # En düşük maliyetli hücreyi bul
            min_cost = float('inf')
            min_row, min_col = -1, -1
            
            for row in unassigned_rows:
                for col in unassigned_cols:
                    if cost_matrix[row, col] < min_cost:
                        min_cost = cost_matrix[row, col]
                        min_row, min_col = row, col
            
            # Eğer minimum maliyet sonsuzsa (eşik değerinden düşük IoU), dur
            if min_cost == float('inf'):
                break
            
            # Atama yap
            assignments.append((min_row, min_col))
            
            # Atanan satır ve sütunu kaldır
            unassigned_rows.remove(min_row)
            unassigned_cols.remove(min_col)
        
        return assignments
    
    def _is_in_pool_area(self, bbox):
        """
        Verilen bounding box'ın havuz alanında olup olmadığını kontrol eder.
        Bu metod ileride ana programdan pool_area veya pool_polygon parametresi alarak güncellenebilir.
        
        Args:
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            
        Returns:
            bool: Havuz alanında ise True, değilse False
        """
        # Şu an için daima False döndür, çünkü havuz alanı ana programda kontrol ediliyor
        return False
    
    def get_tracking_info(self, obj_id):
        """
        Belirli bir nesnenin takip bilgilerini döndürür.
        
        Args:
            obj_id (int): Nesne ID'si
            
        Returns:
            dict: Takip bilgileri sözlüğü veya nesne takip edilmiyorsa None
        """
        if obj_id in self.tracked_objects:
            return self.tracked_objects[obj_id]
        return None
    
    def get_all_tracked_objects(self):
        """
        Tüm takip edilen nesnelerin bilgilerini döndürür.
        
        Returns:
            dict: Nesne ID'lerini anahtar olarak içeren takip bilgileri sözlüğü
        """
        return self.tracked_objects
    
    def reset(self):
        """
        Takip sistemini sıfırlar.
        """
        self.next_id = 0
        self.tracked_objects = {}
        if self.use_kalman:
            self.kalman_filters = {} 

    def _update_all_tracked_objects(self, detections):
        """
        Tüm tespit edilen nesneleri takip listesine günceller.
        
        Args:
            detections: (id, bbox, confidence) formatında tespit listesi
        """
        # Tüm takip edilenlerin görünmeme sayacını artır
        for obj_id in list(self.tracked_objects.keys()):
            self.tracked_objects[obj_id]['disappeared'] += 1
        
        # Hiç tespit yoksa, takip durumunu güncelle ve çık
        if not detections:
            # Çok uzun süre görünmeyen nesneleri sil
            for obj_id in list(self.tracked_objects.keys()):
                if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
                    if self.use_kalman and obj_id in self.kalman_filters:
                        del self.kalman_filters[obj_id]
                    if obj_id in self.movement_data:
                        del self.movement_data[obj_id]
            return
        
        # Mevcut tespitleri takip edilen nesnelerle eşleştir
        if self.tracked_objects:
            # Mevcut takip edilen nesnelerin ID'leri ve kutuları
            tracked_ids = list(self.tracked_objects.keys())
            tracked_boxes = [self.tracked_objects[obj_id]['bbox'] for obj_id in tracked_ids]
            
            # Yeni tespit kutuları
            new_boxes = [bbox for _, bbox, _ in detections]
            
            # IoU matrisini hesapla
            iou_matrix = np.zeros((len(tracked_boxes), len(new_boxes)))
            for i, tracked_box in enumerate(tracked_boxes):
                for j, new_box in enumerate(new_boxes):
                    iou_matrix[i, j] = self._calculate_iou(tracked_box, new_box)
            
            # Eşleştirme için greedy yaklaşım
            matched_indices = self._greedy_assignment(iou_matrix, threshold=self.iou_threshold)
            
            # Eşleşen kutuları güncelle
            for i, j in matched_indices:
                tracked_id = tracked_ids[i]
                _, bbox, conf = detections[j]
                
                # Takip edilen nesneyi güncelle
                self._update_tracked_object(tracked_id, bbox, conf)
            
            # Eşleşmeyen indeksleri bul
            matched_row_indices = [i for i, _ in matched_indices]
            matched_col_indices = [j for _, j in matched_indices]
            
            # Eşleşmemiş takip edilen nesneler (kaybolan nesneler)
            for i, obj_id in enumerate(tracked_ids):
                if i not in matched_row_indices:
                    # Görünmeme sayacını artır (zaten yukarıda yapıldı)
                    # Kalman filtresi ile tahmin yap
                    if self.use_kalman and obj_id in self.kalman_filters:
                        kf = self.kalman_filters[obj_id]
                        predicted_bbox = kf.predict()
                        # Tahmin edilen bbox'ı kullan
                        self.tracked_objects[obj_id]['bbox'] = predicted_bbox
            
            # Eşleşmemiş yeni tespitler (yeni nesneler)
            for j, (_, bbox, conf) in enumerate(detections):
                if j not in matched_col_indices:
                    # Yeni nesneyi takip listesine ekle
                    self._register_new_object(bbox, conf)
        else:
            # Takip edilen nesne yoksa, tüm tespitleri yeni nesne olarak ekle
            for _, bbox, conf in detections:
                self._register_new_object(bbox, conf)
        
        # Çok uzun süre görünmeyen nesneleri sil
        for obj_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                del self.tracked_objects[obj_id]
                if self.use_kalman and obj_id in self.kalman_filters:
                    del self.kalman_filters[obj_id]
                if obj_id in self.movement_data:
                    del self.movement_data[obj_id] 

    def get_movement_data(self, person_id):
        """
        Belirli bir kişi için hareket verilerini döndürür.
        
        Args:
            person_id (int): Kişi ID'si
            
        Returns:
            list: Merkez noktalarının listesi
        """
        if person_id in self.movement_data:
            return self.movement_data[person_id]
        else:
            # Kişi için yeni hareket verisi oluştur
            self.movement_data[person_id] = []
            return [] 