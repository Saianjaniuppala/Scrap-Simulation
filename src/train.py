# src/train.py
import os
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from src.model import get_resnet18

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data', help='data folder with train/val subfolders')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='models/resnet18_finetuned.pth')
    return p.parse_args()

def main():
    args = parse_args()
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    # transforms
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)
    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Train E{epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # validation
        model.eval()
        correct = 0
        total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        acc = correct / total
        print(f"Epoch {epoch+1}/{args.epochs} Loss {running_loss/len(train_ds):.4f} Val Acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            os.makedirs(Path(args.out).parent, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': train_ds.classes
            }, args.out)
            print(f"Saved best model to {args.out}")

    # final classification report
    print(classification_report(y_true, y_pred, target_names=train_ds.classes))

if __name__ == "__main__":
    main()
