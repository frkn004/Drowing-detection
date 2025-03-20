#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nesne Takip Modülü
----------------
Tespit edilen nesneleri takip etmek için kullanılan sınıflar.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2


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