"""
ASL Alphabet Recognition CNN – Letters A–E
Transfer learning (ResNet18) + custom head, augmentation,
label smoothing, cosine LR schedule, and TTA.
"""

import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────
# 1.  Configuration
# ──────────────────────────────────────────────
DATA_DIR     = "asl-dataset"
MODEL_SAVE   = "best_asl_cnn.pth"
IMG_SIZE     = 96       # 128→96 saves ~44% compute per batch
BATCH_SIZE   = 64
NUM_EPOCHS   = 20       
LR           = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
VAL_SPLIT    = 0.2
NUM_WORKERS  = 0
FREEZE_WARMUP= 3
PATIENCE     = 7        # stop a little sooner if plateaued
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────
# 2.  Transforms
# ──────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]

# ──────────────────────────────────────────────
# 3.  Dataset
# ──────────────────────────────────────────────
def make_stratified_split(dataset, val_fraction=0.2, seed=42):
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    train_idx, val_idx = [], []
    rng = random.Random(seed)
    for indices in class_indices.values():
        rng.shuffle(indices)
        split = int(len(indices) * val_fraction)
        val_idx.extend(indices[:split])
        train_idx.extend(indices[split:])
    return train_idx, val_idx


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
CLASSES = full_dataset.classes
NUM_CLASSES = len(CLASSES)
print(f"Classes: {CLASSES}  ({NUM_CLASSES} classes)")

train_idx, val_idx = make_stratified_split(full_dataset, VAL_SPLIT)
train_subset = TransformSubset(Subset(full_dataset, train_idx), train_transform)
val_subset   = TransformSubset(Subset(full_dataset, val_idx),   val_transform)

labels_train   = [full_dataset.targets[i] for i in train_idx]
class_counts   = np.bincount(labels_train)
sample_weights = [1.0 / class_counts[l] for l in labels_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_subset)}  |  Val: {len(val_subset)}")

# ──────────────────────────────────────────────
# 4.  Model — ResNet18 backbone + custom head
#
#  WHY ResNet18?
#    Residual (skip) connections prevent vanishing gradients, letting the
#    network learn effectively without degradation across 18 layers.
#    Pretraining on 1.2M ImageNet images gives strong low-level feature
#    detectors (edges, curves, textures) that transfer directly to hand shapes.
#
#  WHY a custom head instead of the original FC?
#    ResNet18's default FC maps 512→1000 (ImageNet classes). We replace it
#    with a 512→256→5 head:
#      - 256-unit layer: adds a learned non-linear projection before logits,
#        giving more expressive power for the 5-class task.
#      - BatchNorm1d: stabilises training of the new head when the backbone
#        is frozen during warm-up (Phase 1).
#      - Dropout(0.4): prevents co-adaptation of neurons, reduces overfitting
#        on the ~480 training images per class.
#      - Final Linear(256→5): one output neuron per class (A–E).
#
#  WHY no Softmax in forward()?
#    Softmax is applied at inference only (via torch.softmax). During training
#    we use raw logits with LabelSmoothingCE, which calls log_softmax
#    internally — numerically more stable than softmax → log.
# ──────────────────────────────────────────────
class ASLNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Load ResNet18 with pretrained ImageNet weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features  # 512 feature channels out of backbone
        backbone.fc = nn.Identity()            # remove original 1000-class FC layer
        self.backbone = backbone

        # Custom 2-layer classification head
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),  # compress 512 → 256
            nn.BatchNorm1d(256),          # stabilise head training during frozen warm-up
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),              # regularise to reduce overfitting
            nn.Linear(256, num_classes),  # 256 → 5 class logits
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        # Phase 1: only head trains — protects pretrained features from large gradients
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        # Phase 2: fine-tune entire network at reduced LR
        for p in self.backbone.parameters():
            p.requires_grad = True


model = ASLNet(num_classes=NUM_CLASSES).to(DEVICE)

# ──────────────────────────────────────────────
# 5.  Loss / optimizer / scheduler
# ──────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        n = pred.size(1)
        log_p = nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.smoothing / (n - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-smooth * log_p).sum(dim=1).mean()

criterion = LabelSmoothingCE(LABEL_SMOOTH)

model.freeze_backbone()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ──────────────────────────────────────────────
# 6.  Training loop
# ──────────────────────────────────────────────
history = {k: [] for k in ["train_loss","train_acc","val_loss","val_acc","train_f1","val_f1"]}
best_val_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
patience_counter = 0

print("\n" + "="*60)
print(" Starting training")
print("="*60)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    if epoch == FREEZE_WARMUP + 1:
        print(f"\n[Epoch {epoch}] Unfreezing backbone")
        model.unfreeze_backbone()
        optimizer = optim.AdamW(model.parameters(), lr=LR * 0.3, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - FREEZE_WARMUP, eta_min=1e-6)

    # Train
    model.train()
    t_loss, correct, total = 0.0, 0, 0
    all_p, all_l = [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item() * imgs.size(0)
        p = out.argmax(1)
        correct += (p == labels).sum().item()
        total += imgs.size(0)
        all_p.extend(p.cpu().numpy()); all_l.extend(labels.cpu().numpy())

    # Validate
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    v_p, v_l = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            v_loss += criterion(out, labels).item() * imgs.size(0)
            p = out.argmax(1)
            v_correct += (p == labels).sum().item()
            v_total += imgs.size(0)
            v_p.extend(p.cpu().numpy()); v_l.extend(labels.cpu().numpy())

    scheduler.step()

    ta, va = correct/total, v_correct/v_total
    tf, vf = f1_score(all_l,all_p,average="macro"), f1_score(v_l,v_p,average="macro")
    history["train_loss"].append(t_loss/total);  history["val_loss"].append(v_loss/v_total)
    history["train_acc"].append(ta);             history["val_acc"].append(va)
    history["train_f1"].append(tf);              history["val_f1"].append(vf)

    print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
          f"Train Acc={ta:.4f} F1={tf:.4f}  |  "
          f"Val Acc={va:.4f} F1={vf:.4f}  [{time.time()-t0:.1f}s]")

    if va > best_val_acc:
        best_val_acc = va
        best_wts = copy.deepcopy(model.state_dict())
        torch.save(best_wts, MODEL_SAVE)
        patience_counter = 0
        print(f"  ✓ Best val acc: {best_val_acc:.4f} — saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ──────────────────────────────────────────────
# 7.  Final evaluation + TTA
# ──────────────────────────────────────────────
model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
model.eval()

v_p, v_l = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        v_p.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
        v_l.extend(labels.numpy())

print("\nClassification Report (standard):")
print(classification_report(v_l, v_p, target_names=CLASSES))

class RawImageFolder(datasets.ImageFolder):
    def __getitem__(self, i):
        path, target = self.samples[i]
        return self.loader(path), target

raw_ds  = RawImageFolder(root=DATA_DIR)
val_raw = Subset(raw_ds, val_idx)
tta_p, tta_l = [], []
for img_pil, label in val_raw:
    logits = [torch.softmax(model(t(img_pil).unsqueeze(0).to(DEVICE)), 1)
              for t in tta_transforms]
    tta_p.append(torch.stack(logits).mean(0).argmax(1).item())
    tta_l.append(label)

print("\nClassification Report (TTA):")
print(classification_report(tta_l, tta_p, target_names=CLASSES))
tta_acc = np.mean(np.array(tta_p) == np.array(tta_l))
tta_f1  = f1_score(tta_l, tta_p, average="macro")
print(f"TTA Val Accuracy: {tta_acc:.4f}  |  TTA Macro F1: {tta_f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(tta_l, tta_p))

# ──────────────────────────────────────────────
# 8.  Training curves
# ──────────────────────────────────────────────
er = range(1, len(history["train_acc"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ASL A–E CNN Training Curves", fontsize=14, fontweight="bold")
for ax, (tr, vl), title in zip(axes,
    [("train_acc","val_acc"),("train_loss","val_loss"),("train_f1","val_f1")],
    ["Accuracy","Loss","Macro F1"]):
    ax.plot(er, history[tr], label="Train", lw=2)
    ax.plot(er, history[vl], label="Val",   lw=2, ls="--")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nTraining curves → training_curves.png")

# ──────────────────────────────────────────────
# 9.  Inference helpers
# ──────────────────────────────────────────────
def predict_single(image_path: str, use_tta: bool = True) -> str:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if use_tta:
        logits = [torch.softmax(model(t(img).unsqueeze(0).to(DEVICE)), 1)
                  for t in tta_transforms]
        return CLASSES[torch.stack(logits).mean(0).argmax(1).item()]
    return CLASSES[model(val_transform(img).unsqueeze(0).to(DEVICE)).argmax(1).item()]


def predict_folder(folder_path: str):
    from pathlib import Path
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    results = {}
    for p in sorted(Path(folder_path).rglob("*")):
        if p.suffix.lower() in exts:
            pred = predict_single(str(p))
            results[str(p)] = pred
            print(f"  {p.name:40s}  →  {pred}")
    return results


print("\n" + "="*60)
print(f" Best val acc : {best_val_acc:.4f}")
print(f" TTA val acc  : {tta_acc:.4f}")
print(f" TTA macro F1 : {tta_f1:.4f}")
print("="*60)