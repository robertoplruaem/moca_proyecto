import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from tqdm.notebook import tqdm

class CustomOCR_CNN(nn.Module):
    def __init__(self, num_classes=34): # Corregido a 34 clases
        super(CustomOCR_CNN, self).__init__()
        
        # Entrada esperada: 3x128x128
        
        # Bloque Convolucional 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 32x64x64
        )        
        
        # Bloque Convolucional 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 64x32x32
        )       

        # Bloque Convolucional 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 128x16x16
        )
        
        # NUEVO: Bloque Convolucional 4 (Control de dimensionalidad)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Salida: 256x8x8
        )
        
        # Clasificador (Fully Connected)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), # Perfectamente adaptado al nuevo bloque
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x