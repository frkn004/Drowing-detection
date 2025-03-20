#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse

# Global değişkenler
drawing = False  # Seçim yapılıyor mu
ix, iy = -1, -1  # Başlangıç noktası
points = []      # Çokgen noktaları
selection = []   # Seçilen alan [x1, y1, x2, y2]

def draw_rectangle(event, x, y, flags, param):
    """Fare olaylarını yakala ve dikdörtgen çiz"""
    global ix, iy, drawing, selection, img, original_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Sol tuşa basıldığında seçim başlatılır
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        # Fare hareket ederken seçim gösterilir
        if drawing:
            img_copy = original_img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            img = img_copy
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Sol tuş bırakıldığında seçim tamamlanır
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        selection = [x1, y1, x2, y2]
        
        # Seçilen alanı çiz ve bilgileri göster
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"Havuz Alanı: [{x1}, {y1}, {x2}, {y2}]"
        cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    """Ana fonksiyon"""
    global img, original_img, selection
    
    # Selection başlangıçta boş bir liste olarak tanımlanmalı
    selection = []
    
    parser = argparse.ArgumentParser(description="Havuz alanını seçmek için araç")
    parser.add_argument("--image", default="frames/frame_100.jpg", help="Kullanılacak görüntü dosyası")
    parser.add_argument("--output", default="havuz_alani.txt", help="Çıktı dosyası")
    args = parser.parse_args()
    
    # Görüntüyü yükle
    original_img = cv2.imread(args.image)
    if original_img is None:
        print(f"Hata: {args.image} dosyası okunamadı.")
        return
    
    img = original_img.copy()
    height, width = img.shape[:2]
    
    # Pencere oluştur ve fare olaylarını bağla
    window_name = "Havuz Alanını Seçin - (Sol tuş: Seç, 's': Kaydet, 'r': Sıfırla, 'q': Çık)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)
    
    # Başlangıç talimatları
    instruction_img = img.copy()
    text_lines = [
        "Fare ile havuz alanını seçmek için:",
        "1. Sol tuşa basılı tutarak dikdörtgen çizin",
        "2. 's' tuşuna basarak kaydedin",
        "3. 'r' tuşuna basarak yeniden başlayın",
        "4. 'q' tuşuna basarak çıkın"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(instruction_img, line, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
    
    img = instruction_img.copy()
    
    # Ana döngü
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Çık
            break
        elif key == ord('s') and selection:
            # Seçimi kaydet
            with open(args.output, 'w') as f:
                f.write(','.join(map(str, selection)))
            print(f"Havuz alanı kaydedildi: {selection}")
            print(f"Komut satırı için: --pool_area \"{','.join(map(str, selection))}\"")
            break
        elif key == ord('r'):
            # Sıfırla
            img = original_img.copy()
            selection = []
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 