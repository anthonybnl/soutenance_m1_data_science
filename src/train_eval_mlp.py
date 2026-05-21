import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
)

from pathlib import Path

BASE_PATH = Path(__file__).parents[1]

# architecture du modèle


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            *[
                # entrée -> 128
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                # 128 -> 64
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                # 64 -> 32
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                # 32 -> 1
                nn.Linear(32, 1),
            ]
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# dataset


class CustomImageDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.to_numpy()
        self.y = y.to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# méthodes


def _train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, BATCH_SIZE):
    # entraînement d'une époque
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.float().to(DEVICE)
        y_batch = y_batch.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_batch)

    train_loss /= len(train_loader.dataset) - (len(train_loader.dataset) % BATCH_SIZE)
    return train_loss


def _eval_one_epoch(
    model,
    test_loader,
    criterion,
    DEVICE,
    THRESHOLD=0.5,
):
    # évaluation d'une époque : loss, f1, recall, auc
    model.eval()
    val_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.float().to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * len(y_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

    val_loss /= len(test_loader.dataset)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= THRESHOLD).astype(int)
    val_f1 = f1_score(all_labels, preds, zero_division=0)
    val_recall = recall_score(all_labels, preds, zero_division=0)
    val_auc = roc_auc_score(all_labels, all_probs)

    return val_loss, val_f1, val_recall, val_auc


def _boucle_d_entrainement(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    scheduler,
    DEVICE,
    BATCH_SIZE,
    N_EPOCHS=60,
):
    THRESHOLD = 0.5

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_recall": [],
        "val_auc": [],
    }

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, BATCH_SIZE
        )

        val_loss, val_f1, val_recall, val_auc = _eval_one_epoch(
            model, test_loader, criterion, DEVICE, THRESHOLD
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)
        history["val_auc"].append(val_auc)

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1: {val_f1:.4f} | Recall: {val_recall:.4f} | ROC-AUC: {val_auc:.4f}"
        )


def entrainer_modele_mlp(X_train, y_train, X_test, y_test, DEVICE):
    print("Entraînement du modèle MLP depuis zéro ...")

    torch.manual_seed(17)
    generator = torch.Generator()
    generator.manual_seed(17)

    train_dataset = CustomImageDataset(X_train, y_train)
    test_dataset = CustomImageDataset(X_test, y_test)

    BATCH_SIZE = 256

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=generator
    )

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    INPUT_DIM = X_train.shape[1]

    model = MLP(input_dim=INPUT_DIM)

    print(f"Device : {DEVICE}")

    model = model.to(DEVICE)
    pos_weight = pos_weight.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Réduction du LR si la val_loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    _boucle_d_entrainement(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        scheduler,
        DEVICE,
        BATCH_SIZE,
        N_EPOCHS=60,
    )

    # sauvegarde du modele

    MODELS_PATH = BASE_PATH / "models"
    MODELS_PATH.mkdir(exist_ok=True)

    this_model_path = MODELS_PATH / "mlp_model_state_dict.pt"

    # Sauvegarde poids PyTorch
    torch.save(model.state_dict(), this_model_path)

    print(f"Modèle sauvegardé dans : {this_model_path}")

    _print_classification_report(model, test_loader, DEVICE)

    return model


def _print_classification_report(model, test_loader, DEVICE, THRESHOLD=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.float().to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    y_pred = (all_probs >= THRESHOLD).astype(int)

    print("=" * 50)
    print("RAPPORT DE CLASSIFICATION — MLP PyTorch")
    print("=" * 50)
    print(classification_report(all_labels, y_pred, target_names=["Fidèle", "Churner"]))
    print(f"ROC-AUC : {roc_auc_score(all_labels, all_probs):.4f}")


def charger_modele_mlp(DEVICE):
    MODELS_PATH = BASE_PATH / "models"
    MODELS_PATH.mkdir(exist_ok=True)

    this_model_path = MODELS_PATH / "mlp_model_state_dict.pt"

    model = MLP(input_dim=55)

    # on charge les paramètres
    loaded_state_dict = torch.load(this_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(loaded_state_dict)

    model = model.to(DEVICE)

    model.eval()

    return model
