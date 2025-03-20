#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Eğitim Modülü
-----------------
Boğulma tespiti için özel bir model eğitir.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Ana dizini ekle (config modülüne erişim için)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import load_config


class DrowningDataset(Dataset):
    """Boğulma veri kümesi sınıfı."""
    
    def __init__(self, features, labels):
        """
        DrowningDataset sınıfını başlatır.
        
        Args:
            features (numpy.ndarray): Özellik vektörleri
            labels (numpy.ndarray): Etiketler (0: normal, 1: boğulma)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DrowningDetectorModel(nn.Module):
    """Boğulma tespiti için sinir ağı modeli."""
    
    def __init__(self, input_size, hidden_size=64):
        """
        DrowningDetectorModel sınıfını başlatır.
        
        Args:
            input_size (int): Giriş özelliğinin boyutu
            hidden_size (int): Gizli katman boyutu
        """
        super(DrowningDetectorModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        İleri yönlü geçiş.
        
        Args:
            x (torch.Tensor): Giriş tensörü
            
        Returns:
            torch.Tensor: Tahmin tensörü
        """
        return self.model(x)


def load_data(data_path, balance_data=True):
    """
    Veriyi dosyadan yükler.
    
    Args:
        data_path (str): CSV veri dosyasının yolu
        balance_data (bool): Veriyi dengelemek için altörnekleme yapılsın mı
        
    Returns:
        tuple: (features, labels) tuple'ı
    """
    # CSV verilerini yükle
    print(f"Veri yükleniyor: {data_path}")
    data = pd.read_csv(data_path)
    
    # Eksik değerleri doldur
    data.fillna(0, inplace=True)
    
    # Veri setinde dengesizlik varsa (normal örneklerin sayısı boğulma örneklerinden çok daha fazla olabilir)
    if balance_data:
        drowning_samples = data[data['label'] == 1]
        normal_samples = data[data['label'] == 0]
        
        if len(normal_samples) > len(drowning_samples) * 3:  # Eğer normal örnekler 3 katından fazlaysa
            # Normal örneklerden rastgele alt örnekleme yap
            normal_samples = normal_samples.sample(len(drowning_samples) * 3, random_state=42)
            
            # Veri setini yeniden birleştir
            data = pd.concat([normal_samples, drowning_samples])
        
        print(f"Dengelenmiş veri seti boyutu: {len(data)}")
        print(f"Normal örnekler: {len(data[data['label'] == 0])}")
        print(f"Boğulma örnekleri: {len(data[data['label'] == 1])}")
    
    # Özellikleri ve etiketleri ayır
    # NOT: Özellik sütunlarının isimlerini veri setinize uygun şekilde değiştirin
    feature_columns = ['speed', 'avg_speed', 'acceleration', 'movement_variance', 'stillness_count', 'erratic_count']
    label_column = 'label'
    
    features = data[feature_columns].values
    labels = data[label_column].values
    
    return features, labels


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=30, device='cpu'):
    """
    Modeli eğitir.
    
    Args:
        train_loader (DataLoader): Eğitim veri yükleyicisi
        val_loader (DataLoader): Doğrulama veri yükleyicisi
        model (nn.Module): Eğitilecek model
        criterion: Kayıp fonksiyonu
        optimizer: Optimizasyon algoritması
        num_epochs (int): Dönem sayısı
        device (str): Hesaplama cihazı
        
    Returns:
        dict: Eğitim istatistikleri
    """
    model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # En iyi modeli takip etmek için
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Eğitim)"):
            features, labels = features.to(device), labels.to(device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # İleri yönlü geçiş
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Geri yayılım
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Ortalama eğitim kaybını hesapla
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Doğrulama
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Doğrulama)"):
                features, labels = features.to(device), labels.to(device)
                
                # İleri yönlü geçiş
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                val_loss += loss.item()
                
                # Tahminleri kaydet
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Ortalama doğrulama kaybını hesapla
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Doğrulama metriklerini hesapla
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        val_accuracies.append(accuracy)
        
        # Eğitim istatistiklerini yazdır
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1: {f1:.4f}")
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"En iyi model güncellendi (Epoch {epoch+1})")
    
    # En iyi model durumunu yükle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Eğitim istatistiklerini döndür
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss
    }


def plot_training_history(stats, output_path=None):
    """
    Eğitim geçmişini çizer.
    
    Args:
        stats (dict): Eğitim istatistikleri
        output_path (str, optional): Grafik çıktı dosyasının yolu
    """
    plt.figure(figsize=(12, 5))
    
    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(stats['train_losses'], label='Eğitim Kaybı')
    plt.plot(stats['val_losses'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()
    plt.grid(True)
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(stats['val_accuracies'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.title('Doğrulama Doğruluğu')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Eğitim grafiği kaydedildi: {output_path}")
    
    plt.show()


def main():
    """Ana eğitim fonksiyonu."""
    parser = argparse.ArgumentParser(description="Boğulma Tespiti Model Eğitimi")
    parser.add_argument("--data", type=str, required=True, help="CSV veri dosyasının yolu")
    parser.add_argument("--config", type=str, default="../config/default.yaml", help="Konfigürasyon dosyası")
    parser.add_argument("--output", type=str, default="drowning_detector.pt", help="Model çıktı dosyası")
    parser.add_argument("--epochs", type=int, default=30, help="Eğitim dönemleri sayısı")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=0.001, help="Öğrenme oranı")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Hesaplama cihazı (cuda/cpu)")
    args = parser.parse_args()
    
    # Konfigürasyon yükle
    config = load_config(args.config)
    
    # Veriyi yükle
    features, labels = load_data(args.data)
    print(f"Yüklenen veri boyutu: {len(features)} örnek, {features.shape[1]} özellik")
    
    # Eğitim/doğrulama/test bölünmesi
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Eğitim seti: {len(X_train)} örnek")
    print(f"Doğrulama seti: {len(X_val)} örnek")
    print(f"Test seti: {len(X_test)} örnek")
    
    # Veri yükleyicileri oluştur
    train_dataset = DrowningDataset(X_train, y_train)
    val_dataset = DrowningDataset(X_val, y_val)
    test_dataset = DrowningDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Model oluştur
    input_size = features.shape[1]
    model = DrowningDetectorModel(input_size)
    
    # Kayıp fonksiyonu ve optimizasyon algoritması
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model eğitimi başlatılıyor... (Cihaz: {args.device})")
    print(f"Öğrenme oranı: {args.lr}, Batch boyutu: {args.batch_size}, Dönem sayısı: {args.epochs}")
    
    # Modeli eğit
    stats = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device
    )
    
    # Eğitim grafiğini çiz
    plot_training_history(stats, os.path.splitext(args.output)[0] + "_training.png")
    
    # Test veri kümesi üzerinde değerlendir
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Test Değerlendirmesi"):
            features, labels = features.to(args.device), labels.to(args.device)
            
            outputs = model(features)
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Test metriklerini hesapla
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print("\nTest Sonuçları:")
    print(f"Doğruluk: {accuracy:.4f}")
    print(f"Hassasiyet: {precision:.4f}")
    print(f"Geri Çağırma: {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")
    
    # Modeli kaydet
    torch.save(model, args.output)
    print(f"Model kaydedildi: {args.output}")
    
    # Modelin mimarisini ve test sonuçlarını kaydet
    with open(os.path.splitext(args.output)[0] + "_info.yaml", "w") as f:
        yaml.dump({
            'model_architecture': str(model),
            'input_size': input_size,
            'test_results': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'device': args.device,
                'best_val_loss': float(stats['best_val_loss'])
            }
        }, f, default_flow_style=False)


if __name__ == "__main__":
    main() 