#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet trainer: image (5,H,W) → (1,H,W), MSE on valid pixels.
@Author Chris
@Date 2026
"""

import torch
from torch.utils.data import DataLoader

from constants import *
from model.module import build_device
from resnet.resnet_dataset import ResNetTrainDataset
from resnet.resnet_model import ResNet, save_checkpoint, DEFAULT_CONFIG

EPOCHS = 64


def main():
    device = build_device()
    dataset = ResNetTrainDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = ResNet(**DEFAULT_CONFIG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_pix = 0
        for x, y, valid in loader:
            x, y, valid = x.to(device), y.to(device), valid.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = (valid * (pred - y) ** 2).sum() / (valid.sum() + 1e-8)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * valid.sum().item()
            n_pix += valid.sum().item()
        train_loss = epoch_loss / (n_pix + 1e-8)
        print(f"Epoch {epoch + 1}/{EPOCHS}  train_loss: {train_loss:.6f}")

    os.makedirs(os.path.dirname(RESNET_MODEL_PATH), exist_ok=True)
    save_checkpoint(model, dataset.x_mean, dataset.x_std, dataset.y_mean, dataset.y_std, RESNET_MODEL_PATH)


if __name__ == "__main__":
    main()
