import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def build_csv_from_breakhis(breakhis_root, save_dir="data"):
    rows = []
    all_files = glob.glob(
        os.path.join(breakhis_root, "**", "*.png"),
        recursive=True
    )
    print(f"  Total PNG files mile: {len(all_files)}")

    for path in all_files:
        path_lower = path.lower().replace("\\", "/")
        if "/benign/" in path_lower or "\\benign\\" in path.lower():
            label_id   = 0
            label_name = "benign"
        elif "/malignant/" in path_lower or "\\malignant\\" in path.lower():
            label_id   = 1
            label_name = "malignant"
        else:
            continue

        mag = "40X"
        for m in ["40X", "100X", "200X", "400X"]:
            if m.lower() in path.lower():
                mag = m
                break

        if label_name == "benign":
            text = (f"Benign breast tissue biopsy at {mag} magnification. "
                   f"H&E stained slide. Regular cellular patterns observed. "
                   f"No signs of malignancy.")
        else:
            text = (f"Malignant breast tissue biopsy at {mag} magnification. "
                   f"H&E stained slide. Irregular nuclear morphology observed. "
                   f"Increased mitotic activity present.")

        rows.append({
            "image_path"    : path,
            "report_text"   : text,
            "label"         : label_id,
            "magnification" : mag,
        })

    if len(rows) == 0:
        print("  ERROR: No image!")
        return None, None, None

    df = pd.DataFrame(rows)
    print(f"  Total: {len(df)} | Benign: {(df['label']==0).sum()} | Malignant: {(df['label']==1).sum()}")

    df_small, _ = train_test_split(
        df, test_size=0.80, stratify=df["label"], random_state=42
    )
    train_df, temp_df = train_test_split(
        df_small, test_size=0.30, stratify=df_small["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(save_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(save_dir, "test.csv"),  index=False)

    print(f"  CSV saved → Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
    return train_df, val_df, test_df


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.1
            ),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


class BreastCancerDataset(Dataset):
    def __init__(self, csv_path, split="train", max_len=128):
        self.df        = pd.read_csv(csv_path)
        self.transform = get_transforms(split)
        self.max_len   = max_len
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        print(f"  {split}: {len(self.df)} samples loaded")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        enc = self.tokenizer(
            str(row["report_text"]),
            padding       = "max_length",
            max_length    = self.max_len,
            truncation    = True,
            return_tensors= "pt",
        )
        return {
            "image"         : img,
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(int(row["label"]), dtype=torch.long),
        }


def get_dataloader(csv_path, split="train", batch_size=8):
    ds = BreastCancerDataset(csv_path, split)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = 0,
        pin_memory  = torch.cuda.is_available(),
    )
