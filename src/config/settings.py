#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfigürasyon Modülü
------------------
Sistem konfigürasyonunu yükleyen ve yöneten fonksiyonlar.
"""

import os
import yaml


def load_config(config_path=None):
    """
    Konfigürasyon dosyasını yükler.
    
    Args:
        config_path (str, optional): Konfigürasyon dosyasının yolu
        
    Returns:
        dict: Konfigürasyon parametreleri
    """
    # Varsayılan konfigürasyon
    default_config = {
        # İnsan tespiti parametreleri
        'detection': {
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 30,
        },
        
        # Hareket analizi parametreleri
        'movement': {
            'history_size': 30,
            'stillness_threshold': 3,
            'movement_threshold': 5,
        },
        
        # Boğulma tespiti parametreleri
        'drowning': {
            'stillness_threshold': 15,
            'erratic_threshold': 8,
            'alert_duration': 3,
            'model_type': 'rule_based',
            'model_path': None,
        },
        
        # Video işleme parametreleri
        'video': {
            'width': 640,
            'height': 480,
            'fps': 30,
            'buffer_size': 5,
        },
        
        # Görselleştirme parametreleri
        'visualization': {
            'show_tracking': True,
            'show_stats': True,
            'line_thickness': 2,
            'font_scale': 0.6,
        },
    }
    
    # Eğer konfigürasyon dosyası belirtilmişse yükle
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
                # Varsayılan konfigürasyonu yüklenen ile birleştir
                for section, params in loaded_config.items():
                    if section in default_config:
                        default_config[section].update(params)
                    else:
                        default_config[section] = params
                        
            print(f"Konfigürasyon dosyası yüklendi: {config_path}")
        except Exception as e:
            print(f"Konfigürasyon dosyası yüklenirken hata oluştu: {e}")
            print("Varsayılan konfigürasyon kullanılıyor.")
    else:
        print("Konfigürasyon dosyası bulunamadı, varsayılan konfigürasyon kullanılıyor.")
    
    return default_config


def save_config(config, config_path):
    """
    Konfigürasyonu dosyaya kaydeder.
    
    Args:
        config (dict): Konfigürasyon parametreleri
        config_path (str): Konfigürasyon dosyasının yolu
        
    Returns:
        bool: Başarılı ise True, değilse False
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Konfigürasyon dosyası kaydedildi: {config_path}")
        return True
    except Exception as e:
        print(f"Konfigürasyon dosyası kaydedilirken hata oluştu: {e}")
        return False


def get_default_config_path():
    """
    Varsayılan konfigürasyon dosyasının yolunu döndürür.
    
    Returns:
        str: Konfigürasyon dosyasının yolu
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config_path = os.path.join(script_dir, 'config', 'default.yaml')
    return default_config_path 