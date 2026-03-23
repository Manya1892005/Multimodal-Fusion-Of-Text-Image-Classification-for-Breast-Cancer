import os, sys, subprocess

print("="*50)
print("  Breast Cancer Multimodal — Setup & Run")
print("="*50)


print("\n[1/4] Packages are installing...")
pkgs = [
    "torch", "torchvision", "transformers",
    "scikit-learn", "pandas", "Pillow", "tqdm", "numpy"
]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"])
print("  ✓ Packages done")


print("\n[2/4] To make BreakHis folder ")
print()
print("  Windows example:")
print(r"  C:\Users\YourName\Downloads\BreaKHis_v1\histology_slides\breast")
print()
print("  ( To make a folder in File Explorer ")
print("   copy the path from the above address bar")
print()

path = input(" Path  yahan paste karo: ").strip().strip('"').strip("'")

if not os.path.exists(path):
    print(f"\n  ERROR: Ye path nahi mila:")
    print(f"  {path}")
    print()
    print("  Check karo ki BreakHis extract hua hai ya nahi")
    print("  Download link: kaggle.com/datasets/ambarish/breakhis")
    sys.exit(1)

print(f"  ✓ Path sahi hai!")

print("\n[3/4] CSV files ban rahi hain...")
from dataset import build_csv_from_breakhis
build_csv_from_breakhis(path, save_dir="data")
print("  ✓ data/train.csv, val.csv, test.csv ban gayi")


print("\n[4/4] Model check ho raha hai...")
import torch
from model import BreastCancerMultimodalNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = BreastCancerMultimodalNet(num_classes=2).to(device)
imgs  = torch.randn(2, 3, 224, 224).to(device)
ids   = torch.randint(0, 28996, (2, 128)).to(device)
mask  = torch.ones(2, 128, dtype=torch.long).to(device)

logits, alpha, attn = model(imgs, ids, mask)
print(f"  Output shape: {logits.shape} ✓")
print(f"  Model bilkul theek hai!\n")


print("="*50)
ans = input("  Training shuru karein? (y dabao): ").strip().lower()
if ans == "y":
    import train
    train.main()
else:
    print("\n  Baad mein chalana ho to terminal mein likho:")
    print("  python train.py")
