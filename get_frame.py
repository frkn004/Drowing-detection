#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os

# Frames klasörünü oluştur (yoksa)
os.makedirs('frames', exist_ok=True)

# Videoyu aç
video_path = '/Users/furkansevinc/Downloads/bogulma.mp4'
cap = cv2.VideoCapture(video_path)

# Başlangıçta bir kare al
ret, frame = cap.read()
if ret:
    print(f'Video boyutları: {frame.shape[1]}x{frame.shape[0]}')
    cv2.imwrite('frames/frame_reference.jpg', frame)
    print('Referans kare frames/frame_reference.jpg dosyasına kaydedildi')

# 100. kare civarından bir kare daha al
count = 0
while count < 100 and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

if count == 100:
    cv2.imwrite('frames/frame_100.jpg', frame)
    print('100. kare frames/frame_100.jpg dosyasına kaydedildi')

# 300. kare civarından bir kare daha al
while count < 300 and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

if count == 300:
    cv2.imwrite('frames/frame_300.jpg', frame)
    print('300. kare frames/frame_300.jpg dosyasına kaydedildi')

# Kaynakları serbest bırak
cap.release() 