#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hareket Analizi Modülü
----------------------
Havuzdaki insanların hareketlerini analiz ederek yüzme veya boğulma 
belirtilerini tespit etmek için kullanılan sınıf.
"""

import cv2
import numpy as np
from collections import deque


class MovementAnalyzer:
    """Havuzdaki insanların hareketlerini analiz eden sınıf."""
    
    def __init__(self, config=None, history_size=30):
        """
        MovementAnalyzer sınıfını başlatır.
        
        Args:
            config (dict): Konfigürasyon parametreleri
            history_size (int): Hareket geçmişinin tutulacağı kare sayısı
        """
        self.config = config or {}
        self.history_size = history_size
        
        # Her nesne için konum geçmişini takip etmek için sözlük
        self.position_history = {}
        
        # Her nesne için hareket özelliklerini takip etmek için sözlük
        self.movement_features = {}
        
    def analyze(self, frame, bbox, person_id):
        """
        Verilen kişinin hareketlerini analiz eder.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            person_id (int): Kişi ID'si
            
        Returns:
            dict: Hareket özellikleri (hız, ivme, yön, vb.)
        """
        # Kutu merkezini hesapla
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Kişi için hareket geçmişi yoksa oluştur
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.history_size)
            self.movement_features[person_id] = {
                'speed': deque(maxlen=self.history_size),
                'acceleration': deque(maxlen=self.history_size),
                'direction': deque(maxlen=self.history_size),
                'movement_variance': deque(maxlen=self.history_size),
                'stillness_count': 0,
                'erratic_count': 0,
                'last_update': 0
            }
        
        # Mevcut konumu geçmişe ekle
        self.position_history[person_id].append((center_x, center_y))
        
        # Yeterli geçmiş varsa hareket özelliklerini hesapla
        if len(self.position_history[person_id]) >= 2:
            # Son iki konumu al
            prev_x, prev_y = self.position_history[person_id][-2]
            curr_x, curr_y = self.position_history[person_id][-1]
            
            # Hız (piksel/kare) hesapla
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = np.sqrt(dx**2 + dy**2)
            
            # Yön (radyan) hesapla
            direction = np.arctan2(dy, dx)
            
            # İvme hesapla (önceki hız değeri varsa)
            acceleration = 0
            if len(self.movement_features[person_id]['speed']) > 0:
                prev_speed = self.movement_features[person_id]['speed'][-1]
                acceleration = speed - prev_speed
            
            # Hareket varyansı (son N kare için)
            if len(self.position_history[person_id]) >= 10:
                positions = np.array(list(self.position_history[person_id]))
                var_x = np.var(positions[:, 0])
                var_y = np.var(positions[:, 1])
                movement_variance = np.sqrt(var_x**2 + var_y**2)
            else:
                movement_variance = 0
            
            # Özellik listelerini güncelle
            self.movement_features[person_id]['speed'].append(speed)
            self.movement_features[person_id]['direction'].append(direction)
            self.movement_features[person_id]['acceleration'].append(acceleration)
            self.movement_features[person_id]['movement_variance'].append(movement_variance)
            
            # Hareketsizlik tespiti (düşük hız ve hareket varyansı)
            if speed < 3 and movement_variance < 5:
                self.movement_features[person_id]['stillness_count'] += 1
            else:
                self.movement_features[person_id]['stillness_count'] = max(
                    0, self.movement_features[person_id]['stillness_count'] - 1)
            
            # Düzensiz hareket tespiti (yüksek ivme ve yön değişimi)
            if abs(acceleration) > 10 and len(self.movement_features[person_id]['direction']) >= 3:
                dir_change = abs(self.movement_features[person_id]['direction'][-1] - 
                                self.movement_features[person_id]['direction'][-3])
                if dir_change > np.pi/4:  # 45 dereceden fazla yön değişimi
                    self.movement_features[person_id]['erratic_count'] += 1
            else:
                self.movement_features[person_id]['erratic_count'] = max(
                    0, self.movement_features[person_id]['erratic_count'] - 1)
        
        # Hesaplanan hareket özelliklerini döndür
        return {
            'position': (center_x, center_y),
            'position_history': list(self.position_history[person_id]),
            'speed': np.mean(self.movement_features[person_id]['speed']) if self.movement_features[person_id]['speed'] else 0,
            'avg_speed': np.mean(self.movement_features[person_id]['speed']) if self.movement_features[person_id]['speed'] else 0,
            'acceleration': np.mean(self.movement_features[person_id]['acceleration']) if self.movement_features[person_id]['acceleration'] else 0,
            'movement_variance': np.mean(self.movement_features[person_id]['movement_variance']) if self.movement_features[person_id]['movement_variance'] else 0,
            'stillness_count': self.movement_features[person_id]['stillness_count'],
            'erratic_count': self.movement_features[person_id]['erratic_count'],
        }
    
    def reset(self, person_id=None):
        """
        Hareket geçmişini sıfırlar.
        
        Args:
            person_id (int, optional): Sıfırlanacak kişi ID'si. None ise tüm kişileri sıfırlar.
        """
        if person_id is None:
            self.position_history.clear()
            self.movement_features.clear()
        elif person_id in self.position_history:
            del self.position_history[person_id]
            del self.movement_features[person_id] 