#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import FashionMNIST

import ray
from ray import train
from ray.train import torch as ray_torch
from ray.air import session

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCNN(nn.Module):
    """Simple CNN model for Fashion-MNIST classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def get_datasets(data_dir: str):
    """Prepare Fashion-MNIST datasets"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, val_dataset

def train_func(config: Dict[str, Any]):
    """Ray Train function for CNN training"""
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    model = ray_torch.prepare_model(model)
    
    # Prepare datasets
    train_dataset, val_dataset = get_datasets(config["data_dir"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    train_loader = ray_torch.prepare_data_loader(train_loader)
    val_loader = ray_torch.prepare_data_loader(val_loader)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_accuracy = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
        
        logger.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                   f"Train Acc: {train_accuracy:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Report metrics to Ray
        session.report(metrics)
    
    logger.info("Training completed successfully!") 