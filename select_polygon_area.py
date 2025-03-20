#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse

# Global değişkenler
points = []      # Çokgen köşe noktaları
img = None       # Görüntü
original_img = None  # Orijinal görüntü

def click_event(event, x, y, flags, param):
    """Fare olaylarını yakala ve çokgen noktalarını ayarla"""
    global points, img, original_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Sol tıklama - nokta ekle
        points.append((x, y))
        
        # Noktayı çiz
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        
        # Nokta numarasını göster
        cv2.putText(img, str(len(points)), (x+5, y+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Eğer en az 2 nokta varsa, aralarına çizgi çiz
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
            
        # Eğer en az 3 nokta varsa, çokgeni göster
        if len(points) > 2:
            # Çokgeni kapat (son nokta ile ilk nokta arasında çizgi)
            temp_img = img.copy()
            pts = np.array(points + [points[0]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(temp_img, [pts], True, (0, 255, 0), 2)
            cv2.imshow("Havuz Alanını Çokgen Olarak Seçin", temp_img)
        else:
            cv2.imshow("Havuz Alanını Çokgen Olarak Seçin", img)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Sağ tıklama - son noktayı sil
        if points:
            points.pop()
            # Görüntüyü yeniden çiz
            img = original_img.copy()
            for i, point in enumerate(points):
                cv2.circle(img, point, 5, (0, 255, 255), -1)
                cv2.putText(img, str(i+1), (point[0]+5, point[1]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
            # Noktalar arasına çizgi çiz
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], (0, 255, 0), 2)
                
            # Eğer en az 3 nokta varsa, çokgeni kapat
            if len(points) > 2:
                pts = np.array(points + [points[0]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            cv2.imshow("Havuz Alanını Çokgen Olarak Seçin", img)

def main():
    """Ana fonksiyon"""
    global img, original_img, points
    
    parser = argparse.ArgumentParser(description="Havuz alanını çokgen olarak seçmek için araç")
    parser.add_argument("--image", default="frames/frame_100.jpg", help="Kullanılacak görüntü dosyası")
    parser.add_argument("--output", default="havuz_polygon.txt", help="Çıktı dosyası")
    args = parser.parse_args()
    
    # Görüntüyü yükle
    original_img = cv2.imread(args.image)
    if original_img is None:
        print(f"Hata: {args.image} dosyası okunamadı.")
        return
    
    img = original_img.copy()
    height, width = img.shape[:2]
    
    # Pencere oluştur ve fare olaylarını bağla
    window_name = "Havuz Alanını Çokgen Olarak Seçin"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    # Başlangıç talimatları
    instruction_img = img.copy()
    text_lines = [
        "Fare ile havuz alanını çokgen olarak seçmek için:",
        "1. Sol tıklama: Yeni nokta ekle",
        "2. Sağ tıklama: Son noktayı sil",
        "3. 's' tuşu: Seçimi kaydet",
        "4. 'r' tuşu: Sıfırla",
        "5. 'q' tuşu: Çık"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(instruction_img, line, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
    
    img = instruction_img.copy()
    cv2.imshow(window_name, img)
    
    # Ana döngü
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Çık
            break
        elif key == ord('s') and len(points) >= 3:
            # Çokgeni kaydet (en az 3 nokta olmalı)
            # Noktaları x1,y1,x2,y2,...,xn,yn formatında kaydet
            flat_points = []
            for x, y in points:
                flat_points.extend([x, y])
                
            with open(args.output, 'w') as f:
                f.write(','.join(map(str, flat_points)))
                
            points_str = ','.join(map(str, flat_points))
            print(f"Havuz çokgeni kaydedildi: {points} ({len(points)} nokta)")
            print(f"Komut satırı için: --pool_points \"{points_str}\"")
            break
        elif key == ord('r'):
            # Sıfırla
            points = []
            img = original_img.copy()
            # Talimatları tekrar göster
            instruction_img = img.copy()
            y_offset = 30
            for line in text_lines:
                cv2.putText(instruction_img, line, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            img = instruction_img.copy()
            cv2.imshow(window_name, img)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 