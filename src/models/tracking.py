#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nesne Takip Modülü
----------------
Tespit edilen nesneleri takip etmek için kullanılan sınıflar.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque
import cv2
import time


class CentroidTracker:
    """
    Merkez nokta tabanlı nesne takip sınıfı.
    
    Tespit edilen nesneleri takip etmek için merkez noktalarını kullanır.
    """
    
    def __init__(self, max_disappeared=30):
        """
        CentroidTracker sınıfını başlatır.
        
        Args:
            max_disappeared (int): Bir nesne kaç kare kaybolabilir
        """
        # Sonraki atanacak nesne ID'si
        self.nextObjectID = 0
        
        # Takip edilen nesneleri saklar {ID: centroid}
        self.objects = OrderedDict()
        
        # Kaybolan nesneleri saklar {ID: kayıp kare sayısı}
        self.disappeared = OrderedDict()
        
        # Maksimum kayıp kare sayısı (bu sayıyı aşan nesneler takipten çıkar)
        self.maxDisappeared = max_disappeared
    
    def register(self, centroid):
        """
        Yeni bir nesneyi takip listesine ekler.
        
        Args:
            centroid (tuple): Nesnenin merkez koordinatları (x, y)
        """
        # Yeni nesneyi takip listesine ekle
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        """
        Bir nesneyi takip listesinden çıkarır.
        
        Args:
            objectID (int): Çıkarılacak nesnenin ID'si
        """
        # Takip listesinden nesneyi çıkar
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        """
        Sınırlayıcı kutularla takip bilgilerini günceller.
        
        Args:
            rects (list): [(id, [x1, y1, x2, y2], conf), ...] formatında tespit listesi
            
        Returns:
            OrderedDict: Takip edilen nesnelerin ID'lerini ve merkez noktalarını içeren sözlük
        """
        # Hiç tespit yoksa
        if len(rects) == 0:
            # Tüm takip edilen nesnelerin kayıp sayacını artır
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # Maksimum kayıp sayısını aşan nesneleri takipten çıkar
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            # Güncel takip listesini döndür
            return self.objects
        
        # Mevcut tespit merkezleri için boş liste oluştur
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        
        # ID'ler için liste oluştur
        inputIDs = []
        
        # Her tespit için merkez noktasını hesapla
        for i, (person_id, bbox, conf) in enumerate(rects):
            x1, y1, x2, y2 = bbox
            cX = (x1 + x2) // 2
            cY = (y1 + y2) // 2
            inputCentroids[i] = (cX, cY)
            inputIDs.append(person_id)
        
        # Hiç takip edilen nesne yoksa, tüm tespitleri yeni nesne olarak kaydet
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        
        # Aksi halde mevcut takip ve tespitleri eşleştirmeye çalış
        else:
            # Takip edilen nesnelerin ID'lerini ve merkezlerini al
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Her takip edilen merkez ile tespit edilen merkez arasındaki mesafeyi hesapla
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # Satırlara göre sırala (en küçük mesafeleri bul)
            rows = D.min(axis=1).argsort()
            
            # Sütunlara göre sırala (en yakın tespitleri bul)
            cols = D.argmin(axis=1)[rows]
            
            # Kullanılmış satır ve sütunları takip et
            usedRows = set()
            usedCols = set()
            
            # Eşleşmeleri işle
            for (row, col) in zip(rows, cols):
                # İkisi de kullanılmışsa atla
                if row in usedRows or col in usedCols:
                    continue
                
                # Daha önce takip edilen nesnenin ID'sini al ve merkezi güncelle
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                # Bu satır ve sütunu işaretlemiş ol
                usedRows.add(row)
                usedCols.add(col)
            
            # Kullanılmayan satır ve sütunları bul
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # Eğer takip edilen nesne sayısı >= tespit sayısı ise, bazıları
            # kaybolmuş olabilir, kontrol et
            if D.shape[0] >= D.shape[1]:
                # Eşleşmeyen satırlar için
                for row in unusedRows:
                    # ID'yi al
                    objectID = objectIDs[row]
                    
                    # Kayıp sayacını artır
                    self.disappeared[objectID] += 1
                    
                    # Eğer maksimum kayıp sayısını aştıysa takipten çıkar
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # Değilse, yeni nesneler var demektir
            else:
                # Yeni tespitleri kaydet
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        # Güncel takip listesini döndür
        return self.objects 

class AdvancedTracker:
    """
    ByteTrack yaklaşımını temel alan gelişmiş nesne takip sınıfı.
    Düşük güven skorlu tespitleri de değerlendirerek daha tutarlı takip sağlar.
    """
    
    def __init__(self, max_disappeared=30, iou_threshold=0.3, min_hits=3, max_age=30):
        """
        Takip sistemini başlatır.
        
        Args:
            max_disappeared: Nesne kaybolmadan önce izlenecek maksimum kare sayısı
            iou_threshold: Eşleştirme için minimum IoU değeri
            min_hits: Yeni bir izleyici kaydetmek için gereken minimum hit sayısı
            max_age: Bir izleyicinin maksimum yaşı
        """
        self.next_id = 0
        self.trackers = {}  # {id: track_info}
        
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        
        # Yüksek ve düşük güvenli tespit sınıfları
        self.high_thresh = 0.5
        self.low_thresh = 0.1
        
    def update(self, detections):
        """
        Takip sistemini yeni tespitlerle günceller.
        
        Args:
            detections: Tespit listesi [(id, bbox, confidence), ...]
            
        Returns:
            dict: {id: (bbox, confidence)} formatında aktif izleyiciler
        """
        # Detectionları yüksek ve düşük güven skorlulara ayır
        high_dets = []
        low_dets = []
        
        for det in detections:
            _, bbox, conf = det
            if conf >= self.high_thresh:
                high_dets.append((None, bbox, conf))
            elif conf >= self.low_thresh:
                low_dets.append((None, bbox, conf))
        
        # Trackerları güncelle
        self._update_age()
        
        # Yüksek güvenli tespitlerle ilk eşleştirme
        matched_indices, unmatched_trackers, unmatched_detections = \
            self._associate_detections_to_trackers(high_dets)
            
        # Eşleşen trackerleri güncelle
        for i, j in matched_indices:
            tracker_id = list(self.trackers.keys())[i]
            _, bbox, conf = high_dets[j]
            self._update_tracker(tracker_id, bbox, conf)
        
        # Yüksek güvenli eşleşmeyen tespitlerden yeni trackerlar oluştur
        for j in unmatched_detections:
            _, bbox, conf = high_dets[j]
            self._create_new_tracker(bbox, conf)
        
        # Düşük güvenli tespitlerle ikinci eşleştirme (sadece eşleşmemiş trackerlar için)
        if unmatched_trackers and low_dets:
            tracker_indices = unmatched_trackers
            matched_indices, _, _ = self._associate_detections_to_trackers(
                low_dets, tracker_indices=tracker_indices)
            
            # Eşleşen trackerleri güncelle
            for i, j in matched_indices:
                tracker_id = list(self.trackers.keys())[tracker_indices[i]]
                _, bbox, conf = low_dets[j]
                self._update_tracker(tracker_id, bbox, conf)
        
        # Kayıp (belirli süre görünmeyen) trackerları temizle
        self._remove_dead_trackers()
        
        # Aktif trackerları döndür
        active_trackers = {}
        for track_id, info in self.trackers.items():
            if info['hits'] >= self.min_hits and info['time_since_update'] == 0:
                active_trackers[track_id] = (info['bbox'], info['confidence'])
                
        return active_trackers
        
    def _update_age(self):
        """Tüm izleyicilerin yaşını günceller."""
        for track_id in self.trackers:
            self.trackers[track_id]['time_since_update'] += 1
            self.trackers[track_id]['age'] += 1
    
    def _create_new_tracker(self, bbox, confidence):
        """Yeni izleyici oluşturur."""
        self.trackers[self.next_id] = {
            'bbox': bbox,
            'confidence': confidence,
            'time_since_update': 0,
            'hits': 1,
            'age': 1,
            'center_points': deque(maxlen=30),
            'velocities': deque(maxlen=10)
        }
        
        # Merkez noktayı hesapla ve ekle
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.trackers[self.next_id]['center_points'].append((cx, cy))
        
        self.next_id += 1
    
    def _update_tracker(self, track_id, bbox, confidence):
        """Varolan izleyiciyi günceller."""
        # Önceki konumu al (hız hesaplamak için)
        prev_bbox = self.trackers[track_id]['bbox']
        
        # Güncelleme
        self.trackers[track_id]['bbox'] = bbox
        self.trackers[track_id]['confidence'] = confidence
        self.trackers[track_id]['time_since_update'] = 0
        self.trackers[track_id]['hits'] += 1
        
        # Merkez noktayı ekle
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.trackers[track_id]['center_points'].append((cx, cy))
        
        # Hız hesapla (son saniyelerdeki hareket)
        if prev_bbox:
            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
            prev_cx, prev_cy = (prev_x1 + prev_x2) // 2, (prev_y1 + prev_y2) // 2
            
            vx = cx - prev_cx
            vy = cy - prev_cy
            
            self.trackers[track_id]['velocities'].append((vx, vy))
    
    def _remove_dead_trackers(self):
        """Uzun süre güncellenmeyen izleyicileri kaldırır."""
        ids_to_remove = []
        for track_id, info in self.trackers.items():
            if info['time_since_update'] > self.max_disappeared:
                ids_to_remove.append(track_id)
                
        for track_id in ids_to_remove:
            del self.trackers[track_id]
    
    def _associate_detections_to_trackers(self, detections, tracker_indices=None):
        """
        Tespitleri izleyicilere IoU tabanlı greedy eşleştirme ile ilişkilendirir.
        
        Args:
            detections: Tespit listesi [(id, bbox, confidence), ...]
            tracker_indices: Belirli izleyicilerin indeksleri (None = tümü)
            
        Returns:
            tuple: (matched_indices, unmatched_trackers, unmatched_detections)
        """
        if not self.trackers or not detections:
            return [], list(range(len(self.trackers))), list(range(len(detections)))
            
        tracker_keys = list(self.trackers.keys())
        if tracker_indices is not None:
            tracker_keys = [tracker_keys[i] for i in tracker_indices]
            
        tracker_bboxes = [self.trackers[tid]['bbox'] for tid in tracker_keys]
        detection_bboxes = [d[1] for d in detections]
        
        # IoU matrisini hesapla
        iou_matrix = np.zeros((len(tracker_bboxes), len(detection_bboxes)))
        for i, trk_bbox in enumerate(tracker_bboxes):
            for j, det_bbox in enumerate(detection_bboxes):
                iou_matrix[i, j] = self._calculate_iou(trk_bbox, det_bbox)
        
        # Greedy eşleştirme (Hungarian algoritması daha optimal olabilir)
        matched_indices = []
        
        # Satırlar: izleyiciler, Sütunlar: tespitler
        # En yüksek IoU değerlerine sahip çiftleri eşleştir
        if min(iou_matrix.shape) > 0:
            # IoU matrisindeki en yüksek değere sahip hücreleri bulup eşleştir
            while True:
                # En yüksek IoU değerini bul
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[i, j]
                
                # IoU eşik değeri altındaysa eşleştirmeyi durdur
                if max_iou < self.iou_threshold:
                    break
                    
                # Eşleştirmeyi kaydet
                matched_indices.append((i, j))
                
                # Bu satır ve sütunu matristen çıkar
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0
                
        # Eşleşmeyen izleyiciler ve tespitler
        all_tracker_indices = set(range(len(tracker_keys)))
        all_detection_indices = set(range(len(detections)))
        
        matched_tracker_indices = {i for i, _ in matched_indices}
        matched_detection_indices = {j for _, j in matched_indices}
        
        unmatched_trackers = list(all_tracker_indices - matched_tracker_indices)
        unmatched_detections = list(all_detection_indices - matched_detection_indices)
        
        # Eğer belirli izleyici indeksleri kullanıldıysa, doğru indeksleri dönüştür
        if tracker_indices is not None:
            matched_indices = [(tracker_indices[i], j) for i, j in matched_indices]
            unmatched_trackers = [tracker_indices[i] for i in unmatched_trackers]
        
        return matched_indices, unmatched_trackers, unmatched_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """İki sınırlayıcı kutu arasındaki IoU değerini hesaplar."""
        # Koordinatları al
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Kesişim alanını hesapla
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            # Kesişim yok
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Kutuların alanlarını hesapla
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Birleşim alanını hesapla
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # IoU hesapla
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def get_movement_data(self, track_id):
        """Belirli bir izleyici için hareket verilerini döndürür."""
        if track_id in self.trackers:
            return list(self.trackers[track_id]['center_points'])
        return []
    
    def get_velocity(self, track_id):
        """Belirli bir izleyici için hız vektörünü döndürür."""
        if track_id in self.trackers and self.trackers[track_id]['velocities']:
            # Son 3 hızın ortalamasını al
            velocities = list(self.trackers[track_id]['velocities'])[-3:]
            if velocities:
                avg_vx = sum(v[0] for v in velocities) / len(velocities)
                avg_vy = sum(v[1] for v in velocities) / len(velocities)
                return (avg_vx, avg_vy)
        return (0, 0)
    
    def predict_next_position(self, track_id):
        """Sonraki kare için pozisyon tahmini yapar."""
        if track_id not in self.trackers:
            return None
            
        track = self.trackers[track_id]
        x1, y1, x2, y2 = track['bbox']
        
        # Hızı al
        vx, vy = self.get_velocity(track_id)
        
        # Yeni pozisyonu tahmin et
        new_x1 = int(x1 + vx)
        new_y1 = int(y1 + vy)
        new_x2 = int(x2 + vx)
        new_y2 = int(y2 + vy)
        
        return [new_x1, new_y1, new_x2, new_y2] 