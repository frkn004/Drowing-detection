#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video İşleme Modülü
------------------
Video akışlarını işlemek için yardımcı sınıf ve fonksiyonlar.
"""

import cv2
import numpy as np
import time


class VideoProcessor:
    """Video akışlarını işleyen yardımcı sınıf."""
    
    def __init__(self, buffer_size=5, resize_factor=1.0):
        """
        VideoProcessor sınıfını başlatır.
        
        Args:
            buffer_size (int): İşleme için tampon bellek boyutu
            resize_factor (float): Video karelerini yeniden boyutlandırma faktörü 
                                 (1.0=orijinal boyut, 0.5=yarı boyut)
        """
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.resize_factor = resize_factor
        
    def process_frame(self, frame):
        """
        Video karesini işler.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            
        Returns:
            ndarray: İşlenmiş video karesi
        """
        if frame is None:
            return None
        
        # İlk kare ise zamanı başlat
        if self.start_time is None:
            self.start_time = time.time()
        
        # Kare sayacını artır
        self.frame_count += 1
        
        # Performans hesaplaması
        if self.frame_count % 30 == 0:  # Her 30 karede bir FPS güncelle
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Kareyi tampon belleğe ekle
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame)
        
        # Temel görüntü iyileştirme (isteğe bağlı)
        processed_frame = self._enhance_frame(frame)
        
        return processed_frame
    
    def resize_frame(self, frame, target_size=None):
        """
        Video karesini yeniden boyutlandırır.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            target_size (tuple, optional): Hedef boyut (width, height). 
                                         None ise resize_factor kullanılır.
            
        Returns:
            tuple: (yeniden boyutlandırılmış kare, ölçek faktörü)
        """
        if frame is None:
            return None, 1.0
        
        orig_height, orig_width = frame.shape[:2]
        
        if target_size:
            target_width, target_height = target_size
            scale_factor_w = target_width / orig_width
            scale_factor_h = target_height / orig_height
            scale_factor = min(scale_factor_w, scale_factor_h)
        else:
            scale_factor = self.resize_factor
            target_width = int(orig_width * scale_factor)
            target_height = int(orig_height * scale_factor)
        
        if scale_factor == 1.0:
            return frame, scale_factor
        
        resized_frame = cv2.resize(frame, (target_width, target_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return resized_frame, scale_factor
    
    def scale_bbox_to_original(self, bbox, scale_factor):
        """
        Yeniden boyutlandırılmış karedeki sınırlayıcı kutuyu orijinal kare boyutlarına dönüştürür.
        
        Args:
            bbox (list): [x1, y1, x2, y2] formatında sınırlayıcı kutu
            scale_factor (float): Ölçek faktörü
            
        Returns:
            list: Orijinal boyutlarda [x1, y1, x2, y2] formatında sınırlayıcı kutu
        """
        if bbox is None or scale_factor == 1.0:
            return bbox
        
        x1, y1, x2, y2 = bbox
        x1 = int(x1 / scale_factor)
        y1 = int(y1 / scale_factor)
        x2 = int(x2 / scale_factor)
        y2 = int(y2 / scale_factor)
        
        return [x1, y1, x2, y2]
    
    def _enhance_frame(self, frame):
        """
        Görüntü kalitesini iyileştirir.
        
        Args:
            frame (ndarray): İşlenecek video karesi
            
        Returns:
            ndarray: İyileştirilmiş video karesi
        """
        # Temel görüntü iyileştirme (örnek)
        # Gerçek bir projede havuzun özelliklerine göre özelleştirilebilir
        
        # Havuz ortamında daha iyi tespit için kontrast artırma
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def get_background(self):
        """
        Tampon bellek karelerinden arka plan modelini çıkarır.
        
        Returns:
            ndarray: Arka plan modeli karesi
        """
        if not self.frame_buffer:
            return None
        
        # Tampon bellekteki karelerin ortalamasını al
        background = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
        
        return background
    
    def detect_motion(self, frame, background, threshold=25):
        """
        Arka plan çıkarma ile hareket tespiti yapar.
        
        Args:
            frame (ndarray): Mevcut kare
            background (ndarray): Arka plan modeli
            threshold (int): Hareket eşik değeri
            
        Returns:
            ndarray: Hareket maskesi
        """
        if background is None:
            return None
        
        # Mevcut kare ile arka plan arasındaki farkı hesapla
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        
        frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
        background_blur = cv2.GaussianBlur(background_gray, (21, 21), 0)
        
        frame_delta = cv2.absdiff(background_blur, frame_blur)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Gürültü azaltma
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        return thresh
    
    def get_fps(self):
        """
        Mevcut FPS (Frames Per Second) değerini döndürür.
        
        Returns:
            float: FPS değeri
        """
        return self.fps
    
    def reset(self):
        """
        İşleyiciyi sıfırlar.
        """
        self.frame_buffer = []
        self.fps = 0
        self.frame_count = 0
        self.start_time = None 