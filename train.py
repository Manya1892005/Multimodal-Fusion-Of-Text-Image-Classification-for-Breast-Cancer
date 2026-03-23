import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from model   import BreastCancerMultimodalNet
from dataset import get_dataloader


CONFIG = {
    "train_csv"   : "data/train.csv",
    "val_csv"     : "data/val.csv",
    "test_csv"    : "data/test.csv",
    "num_classes" : 2,
    "embed_dim"   : 512,
    "dropout"     : 0.4,
    "batch_size"  : 8,
    "epochs"      : 3,
    "lr"          : 3e-4,
    "bert_lr"     : 2e-5,
    "weight_decay": 5e-5,
    "patience"    : 3,
    "save_path"   : "best_model.pth",
}


def get_class_weights(csv_path, num_classes, device):
    import pandas as pd
    df      = pd.read_csv(csv_path)
    counts  = df["label"].value_counts().sort_index().values.astype(float)
    weights = 1.0 / (counts / counts.sum())
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for batch in tqdm(loader, desc="  Training", leave=False):
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _, _ = model(images, ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, split="Val"):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for batch in tqdm(loader, desc=f"  {split}", leave=False):
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits, _, _ = model(images, ids, mask)
        loss  = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    except Exception:
        auc = 0.0
    return total_loss / len(loader), acc, f1, auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Multimodal Breast Cancer Classifier")
    print(f"  Cross-Modal Attention + Gated Fusion")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    print("Dataloaders ban rahe hain...")
    train_loader = get_dataloader(CONFIG["train_csv"], "train", CONFIG["batch_size"])
    val_loader   = get_dataloader(CONFIG["val_csv"],   "val",   CONFIG["batch_size"])
    test_loader  = get_dataloader(CONFIG["test_csv"],  "test",  CONFIG["batch_size"])

    print("\nModel ban raha hai...")
    model = BreastCancerMultimodalNet(
        num_classes = CONFIG["num_classes"],
        embed_dim   = CONFIG["embed_dim"],
        dropout     = CONFIG["dropout"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    bert_params  = list(model.text_enc.bert.parameters())
    bert_ids     = {id(p) for p in bert_params}
    other_params = [p for p in model.parameters() if id(p) not in bert_ids]

    optimizer = AdamW([
        {"params": other_params, "lr": CONFIG["lr"]},
        {"params": bert_params,  "lr": CONFIG["bert_lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    weights   = get_class_weights(CONFIG["train_csv"], CONFIG["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"\nTraining shuru — {CONFIG['epochs']} epochs\n")
    best_auc, no_improve = 0.0, 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc, tr_f1         = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, va_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train Acc:{tr_acc:.3f} Loss:{tr_loss:.3f} | "
              f"Val Acc:{va_acc:.3f} F1:{va_f1:.3f} AUC:{va_auc:.3f}")

        if va_auc > best_auc:
            best_auc, no_improve = va_auc, 0
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"         ✓ Best model saved (AUC: {best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n{'='*50}")
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))
    _, te_acc, te_f1, te_auc = evaluate(model, test_loader, criterion, device, "Test")
    print(f"  FINAL RESULTS")
    print(f"  Accuracy : {te_acc:.4f} ({te_acc*100:.1f}%)")
    print(f"  F1 Score : {te_f1:.4f}")
    print(f"  ROC-AUC  : {te_auc:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()