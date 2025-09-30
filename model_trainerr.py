from __future__ import annotations
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import re
import csv
import numpy as np

# --- CONFIG ---
DATA_DIR = Path("/kaggle/input/crash-data-binary/binary_format")
VAL_RATIO = 0.4
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New config options
USE_COSINE_SCHEDULER = True  # Toggle cosine annealing scheduler
USE_DATA_AUGMENTATION = True  # Toggle data augmentation
ANNOTATIONS_DIR = Path('/kaggle/input/crashdataset')  # Where *_annotations.csv files are located
MIN_CRASH_SEVERITY = 4  # 1..5: keep only crashed frames with severity >= this threshold; set to 1 to include all
SEVERITY_LOSS_WEIGHT = 1.0  # Weight for severity classification loss
USE_WEIGHTED_SAMPLER = True  # Balance classes by oversampling the minority in training

# --- 1. GET UNIQUE VIDEO NAMES ---
all_images = list((DATA_DIR / "crashed").glob("*.jpg")) + list((DATA_DIR / "normal").glob("*.jpg"))
video_ids = sorted({img.name.split("_crashed_")[0].split("_normal_")[0] for img in all_images})[:3]
print(f"Found {len(video_ids)} unique videos")

# Helpers for annotations
FRAME_NUM_REGEX = re.compile(r"(\d+)(?=\.jpg$)")

def find_annotations_csv(video_id: str) -> Path | None:
    candidates = [
        ANNOTATIONS_DIR / f"{video_id}_annotations.csv",
        DATA_DIR.parent / f"{video_id}_annotations.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def parse_annotations(csv_path: Path) -> list[tuple[int, int, int]]:
    intervals: list[tuple[int, int, int]] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = int(row["frame_start_number"]) if row.get("frame_start_number") is not None else None
                    end = int(row["frame_end_number"]) if row.get("frame_end_number") is not None else None
                    sev = int(row["crash_severity"]) if row.get("crash_severity") is not None else None
                    if start is not None and end is not None and sev is not None:
                        intervals.append((start, end, sev))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return intervals

# Build video -> intervals map
video_to_intervals: dict[str, list[tuple[int, int, int]]] = {}
for vid in video_ids:
    csv_path = find_annotations_csv(vid)
    video_to_intervals[vid] = parse_annotations(csv_path) if csv_path else []


def extract_video_id_and_frame(path: Path) -> tuple[str, int | None]:
    name = path.name
    if "_crashed_" in name:
        video_id = name.split("_crashed_")[0]
    else:
        video_id = name.split("_normal_")[0]
    m = FRAME_NUM_REGEX.search(name)
    frame_num = int(m.group(1)) if m else None
    return video_id, frame_num


def lookup_severity(intervals: list[tuple[int, int, int]], frame_number: int | None) -> int | None:
    if frame_number is None:
        return None
    for start, end, sev in intervals:
        if start <= frame_number <= end:
            return sev
    return None

# --- 2. SPLIT VIDEO IDS ---
random.seed(42)
random.shuffle(video_ids)
val_count = int(len(video_ids) * VAL_RATIO)
val_videos = set(video_ids[:val_count])
train_videos = set(video_ids[val_count:])

print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

# --- 3. CUSTOM DATASET (3 frames sampled from 30-frame sequences) ---
class CrashDataset(Dataset):
    def __init__(self, sequences: list[list[dict]], transform=None):
        # sequences: list of sequences; each sequence is a list of 30 dicts with keys: frame, path, label, severity
        self.sequences = sequences
        self.transform = transform
        self.classes = ["crashed", "normal"]

    def __len__(self):
        return len(self.sequences)

    def _select_three_indices(self, sequence_length: int) -> list[int]:
        # Evenly spaced positions across the sequence
        if sequence_length <= 2:
            return [0] * 3
        return [0, sequence_length // 2, sequence_length - 1]

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        idxs = self._select_three_indices(len(seq))
        selected = [seq[i] for i in idxs]

        imgs = []
        frame_labels = []
        frame_severities = []
        
        # Load all images first
        raw_imgs = []
        for item in selected:
            path: Path = item["path"]
            lbl: int = item["label"]  # 0=crash, 1=normal
            sev: int = int(item["severity"]) if item["severity"] is not None else 0
            img = Image.open(path).convert("RGB")
            raw_imgs.append(img)
            frame_labels.append(lbl)
            frame_severities.append(sev)
        
        # Apply sequence-level transform if available
        if self.transform:
            if isinstance(self.transform, SequenceTransform):
                # Apply the same transform to all frames
                imgs = self.transform(raw_imgs)
            else:
                # Apply individual transforms (for validation)
                imgs = [self.transform(img) for img in raw_imgs]
        else:
            imgs = raw_imgs

        # Stack frames along channel dimension -> (9, H, W)
        x = torch.cat(imgs, dim=0)

        # Determine sample binary label: crash if any frame is crash with severity >= MIN_CRASH_SEVERITY (if threshold > 1)
        def frame_is_positive(lbl: int, sev: int) -> bool:
            if lbl != 0:
                return False
            if MIN_CRASH_SEVERITY is None or MIN_CRASH_SEVERITY <= 1:
                return True
            return sev is not None and sev >= MIN_CRASH_SEVERITY

        any_crash = any(frame_is_positive(l, s) for l, s in zip(frame_labels, frame_severities))
        label = 0 if any_crash else 1

        # Severity is the max severity among the selected frames; 0 if normal
        severity = max([s for l, s in zip(frame_labels, frame_severities) if l == 0], default=0)

        return x, label, severity

# --- 4. BUILD 30-FRAME SEQUENCES ---
# Build video -> frames list with labels and severities (do not filter here to keep 30-length sequences intact)
video_to_frames: dict[str, list[dict]] = {}
for p in all_images:
    video_id, frame_num = extract_video_id_and_frame(p)
    if frame_num is None:
        continue
    label = 0 if "crashed" in p.parts else 1
    sev_val = 0
    if label == 0:
        sev = lookup_severity(video_to_intervals.get(video_id, []), frame_num)
        sev_val = int(sev) if sev is not None else 0
    video_to_frames.setdefault(video_id, []).append({
        "frame": frame_num,
        "path": p,
        "label": label,
        "severity": sev_val,
    })

# Sort frames per video
for vid in list(video_to_frames.keys()):
    video_to_frames[vid].sort(key=lambda d: d["frame"])

# Build non-overlapping 30-frame sequences
all_sequences: list[tuple[str, list[dict]]] = []
for vid, frames in video_to_frames.items():
    if len(frames) < 30:
        continue
    for i in range(0, len(frames) - 29, 30):
        seq = frames[i:i+30]
        all_sequences.append((vid, seq))

# Split sequences by video split
train_sequences = [seq for vid, seq in all_sequences if vid in train_videos]
val_sequences = [seq for vid, seq in all_sequences if vid in val_videos]

# --- SEQUENCE-LEVEL TRANSFORM CLASS ---
class SequenceTransform:
    """Apply the same random transformation to all frames in a sequence"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, imgs):
        if len(imgs) == 0:
            return []
        
        # Capture the current random state
        import random
        import torch
        random_state = random.getstate()
        torch_state = torch.get_rng_state()
        
        # Apply the same transform to all images by resetting the random state
        transformed_imgs = []
        for img in imgs:
            # Reset random state to ensure consistent augmentation
            random.setstate(random_state)
            torch.set_rng_state(torch_state)
            transformed_img = self.transform(img)
            transformed_imgs.append(transformed_img)
        
        return transformed_imgs

# --- 5. TRANSFORMS ---
# Base transform for validation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Training transform with optional data augmentation
if USE_DATA_AUGMENTATION:
    train_transform_base = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_transform = SequenceTransform(train_transform_base)
else:
    train_transform = val_transform

# Keep the original transform variable for backward compatibility
transform = val_transform

# --- 6. DATA LOADERS ---
# Optional: build a weighted sampler to address crash/normal class imbalance in training
def _select_three_indices(sequence_length: int) -> list[int]:
    if sequence_length <= 2:
        return [0] * 3
    return [0, sequence_length // 2, sequence_length - 1]

def _compute_sequence_label(sequence: list[dict]) -> int:
    # Mirror dataset logic: sample 3 evenly-spaced frames, positive if any crash frame meets severity threshold
    idxs = _select_three_indices(len(sequence))
    selected_items = [sequence[i] for i in idxs]

    def frame_is_positive(lbl: int, sev: int) -> bool:
        if lbl != 0:
            return False
        if MIN_CRASH_SEVERITY is None or MIN_CRASH_SEVERITY <= 1:
            return True
        return sev is not None and sev >= MIN_CRASH_SEVERITY

    any_crash = any(frame_is_positive(int(item["label"]), int(item.get("severity", 0))) for item in selected_items)
    return 0 if any_crash else 1

train_sequence_labels = [_compute_sequence_label(seq) for seq in train_sequences]
num_pos = sum(1 for l in train_sequence_labels if l == 0)
num_neg = sum(1 for l in train_sequence_labels if l == 1)

train_sampler = None
if USE_WEIGHTED_SAMPLER and (num_pos > 0 and num_neg > 0):
    # Inverse frequency weighting
    weight_pos = 1.0 / num_pos
    weight_neg = 1.0 / num_neg
    train_sample_weights = [weight_pos if l == 0 else weight_neg for l in train_sequence_labels]
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
elif USE_WEIGHTED_SAMPLER:
    print("[Warning] Skipping weighted sampler because one of the classes has zero samples in training.")

train_dataset = CrashDataset(train_sequences, transform=train_transform)
val_dataset = CrashDataset(val_sequences, transform=val_transform)

if train_sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    print(f"Using WeightedRandomSampler | pos={num_pos}, neg={num_neg}")
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- 7. MODEL ---
class CrashMultiTaskModel(nn.Module):
    def __init__(self, pretrained: bool = True, num_severity_classes: int = 5):
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        # Adjust first conv to accept 9-channel input (3 frames stacked)
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                # Initialize by repeating RGB weights for 3 frames and averaging
                new_conv1.weight[:, 0:3, :, :] = old_conv1.weight.clone()
                new_conv1.weight[:, 3:6, :, :] = old_conv1.weight.clone()
                new_conv1.weight[:, 6:9, :, :] = old_conv1.weight.clone()
                new_conv1.weight /= 3.0
        backbone.conv1 = new_conv1

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.binary_head = nn.Linear(in_features, 1)
        self.severity_head = nn.Linear(in_features, num_severity_classes)

    def forward(self, x):
        feats = self.backbone(x)
        bin_logits = self.binary_head(feats)
        sev_logits = self.severity_head(feats)
        return bin_logits, sev_logits

model = CrashMultiTaskModel(pretrained=True)
model = nn.DataParallel(model)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
severity_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)

# --- 8. SCHEDULER (OPTIONAL) ---
if USE_COSINE_SCHEDULER:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)
else:
    scheduler = None

import matplotlib.pyplot as plt

# --- TRACKING VARIABLES ---
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "train_sev_acc": [],
    "val_sev_acc": [],
    "learning_rate": []
}

# Checkpoint tracking
best_val_acc = 0.0
best_epoch = 0

# --- VISUALIZE SEQUENCE SAMPLES BEFORE TRAINING ---
def visualize_sequence_samples():
    """Visualize sample sequences from train and validation sets"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sequence Samples (3 frames per sequence)', fontsize=16)
    
    # Train samples
    train_sample = train_dataset[0]
    train_imgs, train_label, train_severity = train_sample
    
    # Reshape from (9, H, W) to (3, 3, H, W) for visualization
    train_imgs_reshaped = train_imgs.view(3, 3, IMG_SIZE, IMG_SIZE)
    
    for i in range(3):
        # Convert tensor to numpy for visualization
        img = train_imgs_reshaped[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel(f'Train\nLabel: {"Crash" if train_label == 0 else "Normal"}\nSeverity: {train_severity}', 
                          fontsize=12, rotation=0, ha='right', va='center')
    
    # Validation samples
    val_sample = val_dataset[0]
    val_imgs, val_label, val_severity = val_sample
    
    # Reshape from (9, H, W) to (3, 3, H, W) for visualization
    val_imgs_reshaped = val_imgs.view(3, 3, IMG_SIZE, IMG_SIZE)
    
    for i in range(3):
        # Convert tensor to numpy for visualization
        img = val_imgs_reshaped[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Frame {i+1}')
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel(f'Validation\nLabel: {"Crash" if val_label == 0 else "Normal"}\nSeverity: {val_severity}', 
                          fontsize=12, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    plt.show()

# Visualize samples
visualize_sequence_samples()

# --- TRAINING ---
for epoch in tqdm(range(EPOCHS)):
    # --- Train ---
    model.train()
    train_loss, correct = 0, 0
    train_bce_sum, train_ce_sum = 0, 0
    train_sev_correct, train_sev_total = 0, 0
    for imgs, labels, severities in tqdm(train_loader, total=len(train_loader)):
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)  # 0=crash, 1=normal
        severities = severities.to(DEVICE)  # 0 for normal, 1..5 for crash
        optimizer.zero_grad()
        bin_logits, sev_logits = model(imgs)

        # Binary loss
        bce_loss = criterion(bin_logits, labels)

        # Severity loss only for crash samples with known severity (label == 0 and severity > 0)
        crash_mask = (labels.squeeze(1) == 0) & (severities > 0)
        if crash_mask.any():
            sev_targets = (severities[crash_mask] - 1).long()  # map 1..5 -> 0..4
            ce_loss = severity_criterion(sev_logits[crash_mask], sev_targets)
            # Severity accuracy tracking (train)
            with torch.no_grad():
                sev_pred = sev_logits[crash_mask].argmax(dim=1)
                train_sev_correct += (sev_pred == sev_targets).sum().item()
                train_sev_total += sev_targets.numel()
        else:
            ce_loss = torch.zeros(1, device=DEVICE, dtype=bce_loss.dtype)

        loss = bce_loss + SEVERITY_LOSS_WEIGHT * ce_loss
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        train_loss += loss.item() * batch_size
        train_bce_sum += bce_loss.item() * batch_size
        train_ce_sum += ce_loss.item() * batch_size
        preds = (torch.sigmoid(bin_logits) > 0.5).int()
        correct += (preds == labels.int()).sum().item()

    train_acc = correct / len(train_dataset)
    train_loss /= len(train_dataset)
    train_bce = train_bce_sum / len(train_dataset)
    train_ce = train_ce_sum / len(train_dataset)
    train_sev_acc = (train_sev_correct / train_sev_total) if train_sev_total > 0 else 0.0

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        val_bce_sum, val_ce_sum = 0, 0
        val_sev_correct, val_sev_total = 0, 0
        for imgs, labels, severities in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            severities = severities.to(DEVICE)
            bin_logits, sev_logits = model(imgs)

            bce_loss = criterion(bin_logits, labels)
            crash_mask = (labels.squeeze(1) == 0) & (severities > 0)
            if crash_mask.any():
                sev_targets = (severities[crash_mask] - 1).long()
                ce_loss = severity_criterion(sev_logits[crash_mask], sev_targets)
                # Severity accuracy tracking (val)
                sev_pred = sev_logits[crash_mask].argmax(dim=1)
                val_sev_correct += (sev_pred == sev_targets).sum().item()
                val_sev_total += sev_targets.numel()
            else:
                ce_loss = torch.zeros(1, device=DEVICE, dtype=bce_loss.dtype)

            loss = bce_loss + SEVERITY_LOSS_WEIGHT * ce_loss
            batch_size = imgs.size(0)
            val_loss += loss.item() * batch_size
            val_bce_sum += bce_loss.item() * batch_size
            val_ce_sum += ce_loss.item() * batch_size
            preds = (torch.sigmoid(bin_logits) > 0.5).int()
            val_correct += (preds == labels.int()).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_dataset)
    val_bce = val_bce_sum / len(val_dataset)
    val_ce = val_ce_sum / len(val_dataset)
    val_sev_acc = (val_sev_correct / val_sev_total) if val_sev_total > 0 else 0.0

    # --- Save History ---
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["train_sev_acc"].append(train_sev_acc)
    history["val_sev_acc"].append(val_sev_acc)

    # --- Save Best Checkpoint ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'history': history.copy()
        }, "best_checkpoint.pth")

    # --- Step Scheduler ---
    if scheduler:
        scheduler.step()
    
    # Track learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history["learning_rate"].append(current_lr)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, CE: {train_ce:.4f}) | "
        f"Val Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, CE: {val_ce:.4f}) | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
        f"Train Sev Acc: {train_sev_acc:.4f} | Val Sev Acc: {val_sev_acc:.4f} | "
        f"LR: {current_lr:.2e}"
    )

# --- SAVE LAST CHECKPOINT ---
torch.save({
    'epoch': EPOCHS - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'val_acc': val_acc,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'train_loss': train_loss,
    'history': history
}, "last_checkpoint.pth")

# --- LOAD BEST CHECKPOINT ---
print(f"🏆 Loading best checkpoint from epoch {best_epoch + 1} (Val Acc: {best_val_acc:.4f})")
best_checkpoint = torch.load("best_checkpoint.pth", map_location=DEVICE)
model.load_state_dict(best_checkpoint['model_state_dict'])

# --- SAVE MODEL (keeping original naming for compatibility) ---
torch.save(model.state_dict(), "binary_crash_classifier_video_split.pth")
print("💾 Model saved: binary_crash_classifier_video_split.pth")
print("💾 Best checkpoint saved: best_checkpoint.pth")
print("💾 Last checkpoint saved: last_checkpoint.pth")

# --- PLOT LEARNING CURVE ---
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(18, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch + 1})')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Val Acc")
plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch + 1})')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

# Learning Rate
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history["learning_rate"], label="Learning Rate", color='green')
plt.axvline(x=best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch + 1})')
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()