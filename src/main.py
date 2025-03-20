#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Havuz Güvenlik Sistemi Ana Programı
---------------------------------
Belirtilen video üzerinde insan tespiti ve boğulma analizi yapar.
"""

import os
import cv2
import time
import json
import torch
import numpy as np
import argparse
from ultralytics import YOLO

from models.detection import PersonDetector
from models.tracking import CentroidTracker
from models.drowning import DrowningDetector
from visualization.display import Visualizer


def parse_arguments():
    """
    Komut satırı argümanlarını ayrıştırır.
    
    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(description='Boğulma tespiti için video analizi')
    parser.add_argument('--video_path', type=str, required=True, help='Video dosyası yolu')
    parser.add_argument('--output_path', type=str, default=None, help='Çıktı video dosyası yolu')
    parser.add_argument('--model', type=str, default='yolov8m', help='YOLOv8 model boyutu (n, s, m, l, x)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Tespit eşik değeri')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU eşik değeri')
    parser.add_argument('--area_threshold', type=int, default=100, help='Minimum tespit alanı (piksel cinsinden)')
    parser.add_argument('--max_disappeared', type=int, default=30, help='Maksimum kayboluş sayısı')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Video yeniden boyutlandırma faktörü')
    parser.add_argument('--use_kalman', action='store_true', help='Kalman filtresi kullan')
    parser.add_argument('--pool_area', type=str, default=None, help='Havuz alanı (x1,y1,x2,y2)')
    parser.add_argument('--pool_points', type=str, default=None, help='Havuz alanı poligonu (x1,y1,x2,y2,...,xn,yn)')
    parser.add_argument('--show', action='store_true', help='Görselleştirmeyi göster')
    parser.add_argument('--enhance_pool', action='store_true', help='Havuz tespitini geliştir')
    parser.add_argument('--contrast', type=float, default=1.3, help='Havuz kontrast artırma faktörü')
    parser.add_argument('--brightness', type=int, default=10, help='Havuz parlaklık artırma değeri')
    parser.add_argument('--pool_threshold', type=float, default=0.2, help='Havuz tespiti için eşik değeri')
    return parser.parse_args()


def detect_pool_area(video_path, resize_factor=1.0):
    """
    Video karelerini analiz ederek havuz alanını otomatik tespit eder.
    
    Birkaç kareyi alır, mavi/yeşil rengi tespit eder ve olası havuz alanını bulur.
    
    Args:
        video_path (str): Video dosyasının yolu
        resize_factor (float): Yeniden boyutlandırma faktörü
        
    Returns:
        list: [x1, y1, x2, y2] formatında havuz koordinatları veya None
    """
    try:
        # Video dosyasını aç
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video açılamadı!")
            return None
        
        # Video özelliklerini al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Yeniden boyutlandırılmış boyutlar
        if resize_factor != 1.0:
            width = int(width * resize_factor)
            height = int(height * resize_factor)
        
        # Referans kareyi al (video ortasında bir kare)
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if not ret:
            print("Kare okunamadı!")
            cap.release()
            return None
        
        # Yeniden boyutlandır
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (width, height))
        
        # Referans kareyi HSV'ye dönüştür
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Mavi/yeşil havuzlar için renk aralıkları
        # Açık/koyu mavi renk aralığı
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Yeşilimsi mavi renk aralığı
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Mavi ve yeşil maskeleri oluştur
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # İki maskeyi birleştir
        mask = cv2.bitwise_or(mask_blue, mask_green)
        
        # Gürültüyü azalt
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Referans kareler dizini için "frames" klasörü oluştur
        os.makedirs("frames", exist_ok=True)
        
        # Referans kareyi kaydet
        cv2.imwrite("frames/frame_reference.jpg", frame)
        
        # İlk ve orta kareleri de kaydet
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame_100 = cap.read()
        if ret and resize_factor != 1.0:
            frame_100 = cv2.resize(frame_100, (width, height))
        if ret:
            cv2.imwrite("frames/frame_100.jpg", frame_100)
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame_300 = cap.read()
        if ret and resize_factor != 1.0:
            frame_300 = cv2.resize(frame_300, (width, height))
        if ret:
            cv2.imwrite("frames/frame_300.jpg", frame_300)
        
        cap.release()
        
        # Eğer kontur bulunduysa
        if contours:
            # En büyük kontur seç (muhtemelen havuz)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Minimum çevreleyen dikdörtgeni bul
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Sonucu döndür
            return [x, y, x+w, y+h]
        
        # Kontur bulunamadıysa, varsayılan olarak ekranın ortasını al
        else:
            # Ekranın ortasını merkez alan havuz alanı
            x1 = int(width * 0.25)
            y1 = int(height * 0.25)
            x2 = int(width * 0.75)
            y2 = int(height * 0.75)
            return [x1, y1, x2, y2]
    
    except Exception as e:
        print(f"Havuz alanı tespitinde hata: {str(e)}")
        return None


def create_polygon_from_points(points_str):
    """
    Nokta dizisinden çokgen oluşturur.
    
    Args:
        points_str (str): "x1,y1,x2,y2,...,xn,yn" formatında nokta dizisi
    
    Returns:
        list: [(x1,y1), (x2,y2), ..., (xn,yn)] formatında noktalar
    """
    try:
        if not points_str:
            return None
            
        # Virgülle ayrılmış değerleri parçala
        values = list(map(int, points_str.strip().split(',')))
        
        # Değer sayısı çift olmalı (x,y çiftleri)
        if len(values) % 2 != 0:
            print("Uyarı: Havuz çokgeni noktaları geçersiz format. Çift sayıda değer olmalı.")
            return None
            
        # Koordinat çiftleri oluştur
        points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        
        # En az 3 nokta gerekli (üçgen)
        if len(points) < 3:
            print("Uyarı: Havuz çokgeni en az 3 nokta içermeli.")
            return None
            
        return points
    except Exception as e:
        print(f"Hata: Havuz çokgeni oluşturulamadı: {str(e)}")
        return None


def points_to_rectangle(points):
    """
    Noktalardan sınırlayıcı dikdörtgen oluşturur.
    
    Args:
        points (list): [(x1,y1), (x2,y2), ...] formatında nokta listesi
        
    Returns:
        list: [min_x, min_y, max_x, max_y] formatında dikdörtgen
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def is_point_in_polygon(point, polygon):
    """
    Ray casting algoritması ile bir noktanın çokgen içinde olup olmadığını kontrol eder.
    
    Args:
        point: (x, y) formatında kontrol edilecek nokta
        polygon: [(x1,y1), (x2,y2), ...] formatında çokgen köşe noktaları
        
    Returns:
        bool: Nokta çokgen içindeyse True, değilse False
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def enhance_pool_area(frame, pool_mask, contrast=1.3, brightness=10):
    """
    Havuz alanındaki görüntü kalitesini artırır.
    
    Args:
        frame: Orijinal video karesi
        pool_mask: Havuz alanı maskesi
        contrast: Kontrast artırma faktörü (1.0 = değişiklik yok)
        brightness: Parlaklık artırma değeri (0 = değişiklik yok)
        
    Returns:
        Geliştirilmiş görüntü
    """
    # Orijinal görüntüyü kopyala
    enhanced = frame.copy()
    
    # Havuz maskesi varsa uygula
    if pool_mask is not None:
        # Kontrast ve parlaklık ayarla
        pool_area = cv2.addWeighted(
            enhanced, contrast, 
            np.zeros(enhanced.shape, enhanced.dtype), 0, 
            brightness
        )
        
        # Havuz maskesini 3 kanala genişlet
        if len(pool_mask.shape) == 2:
            mask_3channel = cv2.merge([pool_mask, pool_mask, pool_mask])
        else:
            mask_3channel = pool_mask
            
        # Geliştirilmiş havuz alanını birleştir
        cv2.copyTo(pool_area, mask_3channel, enhanced)
        
    return enhanced


def create_pool_mask(frame_shape, pool_coords=None, pool_polygon=None):
    """
    Havuz alanı için maske oluşturur.
    
    Args:
        frame_shape: Görüntü boyutu (height, width)
        pool_coords: Havuz koordinatları [x1, y1, x2, y2] (dikdörtgen)
        pool_polygon: Havuz poligonu [[x1, y1], [x2, y2], ...]
        
    Returns:
        Havuz maskesi (binary)
    """
    # Boş maske oluştur
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    
    if pool_coords is not None:
        # Dikdörtgen havuz alanı
        x1, y1, x2, y2 = pool_coords
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    elif pool_polygon is not None:
        # Poligon havuz alanı
        points = np.array(pool_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    return mask


def main():
    """Ana fonksiyon."""
    # Argümanları işle
    args = parse_arguments()
    
    # Video dosyasını aç
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı: {args.video_path}")
        return
    
    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.resize_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.resize_factor)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Çıktı video dosyası
    out = None
    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    # Modeli yükle
    model_path = f"yolov8{args.model[-1]}.pt" if args.model.startswith("yolov8") else args.model
    print(f"Model yükleniyor: {model_path}")
    yolo_model = YOLO(model_path)
    
    # Tespit ve takip sistemini oluştur
    detector = PersonDetector(
        model=yolo_model, 
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        area_threshold=args.area_threshold,
        max_disappeared=args.max_disappeared,
        use_kalman=args.use_kalman
    )
    
    # Boğulma tespiti sistemi
    drowning_detector = DrowningDetector()
    
    # Havuz alanını belirle
    pool_coords = None
    pool_polygon = None
    
    if args.pool_area:
        pool_coords = list(map(int, args.pool_area.split(',')))
        # Resize faktörünü uygula
        pool_coords = [int(c * args.resize_factor) for c in pool_coords]
        print(f"Havuz alanı: {pool_coords}")
    
    if args.pool_points:
        points = list(map(int, args.pool_points.split(',')))
        # Resize faktörünü uygula
        points = [int(p * args.resize_factor) for p in points]
        # Noktaları (x, y) koordinat çiftlerine dönüştür
        pool_polygon = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        print(f"Havuz poligonu: {len(pool_polygon)} nokta")
    
    # Havuz maskesi oluştur
    pool_mask = None
    
    # İstatistik değişkenleri
    frame_count = 0
    total_time = 0
    total_detections = 0
    
    # Kalman filtresi aktif mi?
    if args.use_kalman:
        print("Kalman filtresi takip sistemi aktif.")
    
    print(f"İşleme başlanıyor... Toplam kare: {total_frames}")
    
    # Ana işleme döngüsü
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Video karesi boyutunu ayarla
        if args.resize_factor != 1.0:
            frame = cv2.resize(frame, (width, height))
        
        # Kare sayacını artır
        frame_count += 1
        
        # Başlangıç zamanını kaydet
        start_time = time.time()
        
        # İlk karede havuz maskesi oluştur
        if frame_count == 1 and (pool_coords is not None or pool_polygon is not None):
            pool_mask = create_pool_mask(frame.shape, pool_coords, pool_polygon)
        
        # Görüntüyü geliştir
        if args.enhance_pool and pool_mask is not None:
            enhanced_frame = enhance_pool_area(
                frame, 
                pool_mask, 
                contrast=args.contrast, 
                brightness=args.brightness
            )
        else:
            enhanced_frame = frame.copy()
        
        # Tespit işlemi
        detections = []
        if args.enhance_pool and pool_mask is not None:
            # Havuz içinde daha düşük eşikle tespit
            detections = detector.detect_in_pool(
                enhanced_frame, 
                pool_mask=pool_mask, 
                pool_threshold=args.pool_threshold
            )
        else:
            # Normal tespit
            detections = detector.detect(enhanced_frame)
        
        # Toplam tespit sayısını güncelle
        total_detections += len(detections)
        
        # Her bir tespit için boğulma analizi
        for person_id, bbox, conf in detections:
            # Kişinin merkez noktasını hesapla
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Kişinin havuz içinde olup olmadığını kontrol et
            is_in_pool = False
            if pool_coords is not None:
                # Dikdörtgen havuz alanı kontrolü
                px1, py1, px2, py2 = pool_coords
                is_in_pool = px1 <= cx <= px2 and py1 <= cy <= py2
            elif pool_polygon is not None:
                # Poligon havuz alanı kontrolü - manuel yöntem
                is_in_pool = is_point_in_polygon((cx, cy), pool_polygon)
            
            # Hareket verisini güncelle
            movement_data = detector.get_movement_data(person_id)
            
            # Boğulma analizi
            score, alert = drowning_detector.detect(movement_data, person_id, is_in_pool)
            
            # Renkler
            if alert:
                color = (0, 0, 255)  # Kırmızı (Boğulma tehlikesi)
            elif is_in_pool:
                color = (0, 255, 255)  # Sarı (Havuzda)
            else:
                color = (0, 255, 0)  # Yeşil (Normal)
            
            # Sınırlayıcı kutuyu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID ve skor bilgisini çiz
            label = f"ID:{person_id}"
            if is_in_pool:
                label += f" Skor:{score:.2f}"
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Havuz alanını göster
        if pool_coords is not None:
            x1, y1, x2, y2 = pool_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Havuz", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif pool_polygon is not None:
            pts = np.array(pool_polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            if len(pool_polygon) > 0:
                cv2.putText(frame, "Havuz", pool_polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # İşlem süresini hesapla
        end_time = time.time()
        process_time = end_time - start_time
        total_time += process_time
        
        # FPS bilgisini ekle
        fps_text = f"FPS: {1/process_time:.2f}" if process_time > 0 else "FPS: ?"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Görselleştirmeyi göster
        if args.show:
            cv2.imshow('Boğulma Tespiti', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Çıktı video dosyasına yaz
        if out is not None:
            out.write(frame)
        
        # İlerlemeyi göster
        if frame_count % 100 == 0 or frame_count == total_frames:
            avg_time = total_time / frame_count
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            avg_detections = total_detections / frame_count
            
            print(f"İlerleme: {frame_count}/{total_frames} kareler "
                  f"({frame_count/total_frames*100:.1f}%) - "
                  f"Ortalama FPS: {avg_fps:.2f}, "
                  f"İşlem Süresi: {avg_time*1000:.2f} ms, "
                  f"Tespit: {avg_detections:.2f}")
    
    # Kaynakları serbest bırak
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # İstatistikleri yazdır
    if frame_count > 0:
        avg_time = total_time / frame_count
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        avg_detections = total_detections / frame_count
        
        print(f"\nİşlem tamamlandı!")
        print(f"Toplam kareler: {frame_count}")
        print(f"Toplam süre: {total_time:.2f} saniye")
        print(f"Ortalama FPS: {avg_fps:.2f}")
        print(f"Ortalama işlem süresi: {avg_time*1000:.2f} ms")
        print(f"Ortalama tespit sayısı: {avg_detections:.2f}")


if __name__ == "__main__":
    main() 