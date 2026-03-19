"""
ASL Webcam Tester — loads best_asl_cnn.pth directly, no retraining.
"""

import torch
import torch.nn as nn
import cv2
from torchvision import transforms, models

# ── Config (must match what you trained with) ──
MODEL_PATH = "best_asl_cnn.pth"
CLASSES    = ["A", "B", "C", "D", "E"]
IMG_SIZE   = 128
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(f"Loaded model from {MODEL_PATH}")

# ── Transform ──
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Webcam loop ──
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop center square for hand region
    h, w = frame.shape[:2]
    size = min(h, w)
    y1, x1 = (h - size) // 2, (w - size) // 2
    hand_crop = frame[y1:y1+size, x1:x1+size]

    # Draw crop box
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (0, 255, 0), 2)

    # Predict
    rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    label = CLASSES[pred_idx]
    confidence = probs[pred_idx].item()

    # Overlay
    cv2.putText(frame, f"{label}  ({confidence:.0%})",
                (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 0), 3)

    cv2.imshow("ASL Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()