"""
ASL File Tester — loads best_asl_cnn.pth and runs inference on
all images in the "test images" folder in the same directory.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Config (must match what you trained with) ──
MODEL_PATH     = Path(__file__).parent / "best_asl_cnn.pth"
TEST_FOLDER    = Path(__file__).parent / "test images"
CLASSES        = ["A", "B", "C", "D", "E"]
IMG_SIZE       = 96
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Rebuild model architecture & load weights ──
class ASLNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


model = ASLNet(num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Loaded model from {MODEL_PATH}\n")

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Inference ──
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], probs[pred_idx].item(), probs.cpu().tolist()


def print_result(path, label, confidence, probs):
    bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    print(f"  File      : {Path(path).name}")
    print(f"  Prediction: {label}  ({confidence:.1%})")
    print(f"  Confidence: [{bar}]")
    print(f"  All probs : { {CLASSES[i]: f'{p:.1%}' for i, p in enumerate(probs)} }")
    print()


# ── Main ──
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = sorted(p for p in TEST_FOLDER.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
if not image_files:
    print(f"No images found in '{TEST_FOLDER}'")
else:
    print(f"Found {len(image_files)} image(s) in '{TEST_FOLDER}'\n")
    print("=" * 55)

    correct, total = 0, 0
    for img_path in image_files:
        label, confidence, probs = predict(img_path)
        print_result(img_path, label, confidence, probs)

        # Auto-score if subfolders are named after classes (e.g. test images/A/img.jpg)
        if img_path.parent.name.upper() in CLASSES:
            true_label = img_path.parent.name.upper()
            total += 1
            if label == true_label:
                correct += 1

    print("=" * 55)
    if total > 0:
        print(f"\nAuto-scored {total} images with known labels:")
        print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")
    else:
        print(f"\nDone. ({len(image_files)} images predicted)")