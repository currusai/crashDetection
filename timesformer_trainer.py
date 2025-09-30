from __future__ import annotations
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, TimesformerModel

# --- CONFIG ---
DATA_DIR = Path("/kaggle/input/crash-data-binary/binary_format")
VAL_RATIO = 0.4
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New config options (mirroring model_trainerr.py)
USE_COSINE_SCHEDULER = True
USE_DATA_AUGMENTATION = True
ANNOTATIONS_DIR = Path('/kaggle/input/crashdataset')
MIN_CRASH_SEVERITY = 4
SEVERITY_LOSS_WEIGHT = 1.0
USE_WEIGHTED_SAMPLER = True

# TimeSformer-specific
TIMESFORMER_CHECKPOINT = "facebook/timesformer-base-finetuned-k400"
NUM_FRAMES = 8  # number of frames sampled per 30-frame sequence

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

# --- 3. BUILD 30-FRAME SEQUENCES (keep semantics from model_trainerr.py) ---
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


# --- 4. TRANSFORMS ---
class SequenceTransform:
    """Apply the same random transformation to all frames in a sequence (PIL in -> PIL out)."""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, imgs):
        if len(imgs) == 0:
            return []
        import random as _random
        import torch as _torch
        random_state = _random.getstate()
        torch_state = _torch.get_rng_state()
        transformed_imgs = []
        for img in imgs:
            _random.setstate(random_state)
            _torch.set_rng_state(torch_state)
            transformed_img = self.transform(img)
            transformed_imgs.append(transformed_img)
        return transformed_imgs


if USE_DATA_AUGMENTATION:
    train_transform_base = nn.Sequential()  # placeholder to keep type hints happy
    # Compose torchvision transforms that keep PIL format (no ToTensor/Normalize here)
    from torchvision import transforms as tvt
    train_transform_base = tvt.Compose([
        tvt.RandomHorizontalFlip(p=0.5),
        tvt.RandomRotation(degrees=10),
        tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tvt.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    train_transform = SequenceTransform(train_transform_base)
else:
    train_transform = None

# Validation: no augmentation; let image_processor handle resize/normalize
val_transform = None


# --- 5. DATASET ---
def _select_evenly_spaced_indices(sequence_length: int, num_frames: int) -> list[int]:
    if sequence_length <= 0:
        return []
    if num_frames <= 1:
        return [0]
    # even spacing across [0, sequence_length-1]
    idxs = np.linspace(0, sequence_length - 1, num=num_frames)
    idxs = np.round(idxs).astype(int).tolist()
    return idxs

def _compute_sequence_label(sequence: list[dict]) -> int:
    idxs = _select_evenly_spaced_indices(len(sequence), NUM_FRAMES)
    selected_items = [sequence[i] for i in idxs]

    def frame_is_positive(lbl: int, sev: int) -> bool:
        if lbl != 0:
            return False
        if MIN_CRASH_SEVERITY is None or MIN_CRASH_SEVERITY <= 1:
            return True
        return sev is not None and sev >= MIN_CRASH_SEVERITY

    any_crash = any(frame_is_positive(int(item["label"]), int(item.get("severity", 0))) for item in selected_items)
    return 0 if any_crash else 1


class CrashVideoDataset(Dataset):
    def __init__(self, sequences: list[list[dict]], transform=None):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        idxs = _select_evenly_spaced_indices(len(seq), NUM_FRAMES)
        selected = [seq[i] for i in idxs]

        pil_frames = []
        frame_labels = []
        frame_severities = []
        for item in selected:
            path: Path = item["path"]
            lbl: int = int(item["label"])  # 0=crash, 1=normal
            sev: int = int(item.get("severity", 0))
            img = Image.open(path).convert("RGB")
            pil_frames.append(img)
            frame_labels.append(lbl)
            frame_severities.append(sev)

        if self.transform:
            pil_frames = self.transform(pil_frames)

        def frame_is_positive(lbl: int, sev: int) -> bool:
            if lbl != 0:
                return False
            if MIN_CRASH_SEVERITY is None or MIN_CRASH_SEVERITY <= 1:
                return True
            return sev is not None and sev >= MIN_CRASH_SEVERITY

        any_crash = any(frame_is_positive(l, s) for l, s in zip(frame_labels, frame_severities))
        label = 0 if any_crash else 1
        severity = max([s for l, s in zip(frame_labels, frame_severities) if l == 0], default=0)

        return pil_frames, label, severity


# --- 6. IMAGE PROCESSOR & COLLATOR ---
image_processor = AutoImageProcessor.from_pretrained(TIMESFORMER_CHECKPOINT)

def collate_videos(batch):
    frames_lists, labels, severities = zip(*batch)  # each frames_list is a list[PIL]
    # image_processor can take list of list of PIL frames to form a batch
    proc = image_processor(list(frames_lists), return_tensors="pt")
    pixel_values = proc["pixel_values"]  # (B, T, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    severities = torch.tensor(severities, dtype=torch.long)
    return pixel_values, labels, severities


# --- 7. DATA LOADERS & SAMPLER ---
train_sequence_labels = [_compute_sequence_label(seq) for seq in train_sequences]
num_pos = sum(1 for l in train_sequence_labels if l == 0)
num_neg = sum(1 for l in train_sequence_labels if l == 1)

train_sampler = None
if USE_WEIGHTED_SAMPLER and (num_pos > 0 and num_neg > 0):
    weight_pos = 1.0 / max(num_pos, 1)
    weight_neg = 1.0 / max(num_neg, 1)
    train_sample_weights = [weight_pos if l == 0 else weight_neg for l in train_sequence_labels]
    train_sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
elif USE_WEIGHTED_SAMPLER:
    print("[Warning] Skipping weighted sampler because one of the classes has zero samples in training.")

train_dataset = CrashVideoDataset(train_sequences, transform=train_transform)
val_dataset = CrashVideoDataset(val_sequences, transform=val_transform)

if train_sampler is not None:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_videos)
    print(f"Using WeightedRandomSampler | pos={num_pos}, neg={num_neg}")
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_videos)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_videos)


# --- 8. MODEL ---
class CrashTimesformerMultiTask(nn.Module):
    def __init__(self, base_ckpt: str, num_severity_classes: int = 5):
        super().__init__()
        self.backbone = TimesformerModel.from_pretrained(base_ckpt)
        hidden_size = self.backbone.config.hidden_size
        self.binary_head = nn.Linear(hidden_size, 1)
        self.severity_head = nn.Linear(hidden_size, num_severity_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # pooled representation: (B, H)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        bin_logits = self.binary_head(pooled)
        sev_logits = self.severity_head(pooled)
        return bin_logits, sev_logits


model = CrashTimesformerMultiTask(TIMESFORMER_CHECKPOINT)
model = nn.DataParallel(model)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
severity_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)

# --- 9. SCHEDULER (OPTIONAL) ---
if USE_COSINE_SCHEDULER:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)
else:
    scheduler = None


# --- 10. TRACKING ---
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "train_sev_acc": [],
    "val_sev_acc": [],
    "learning_rate": []
}

best_val_acc = 0.0
best_epoch = 0


# --- 11. VISUALIZE SEQUENCE SAMPLES BEFORE TRAINING ---
def visualize_sequence_samples():
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("No samples to visualize.")
        return
    fig, axes = plt.subplots(2, min(3, NUM_FRAMES), figsize=(15, 8))
    fig.suptitle('Sequence Samples (first frames shown)', fontsize=16)

    train_frames, train_label, train_sev = train_dataset[0]
    for i in range(min(3, len(train_frames))):
        axes[0, i].imshow(train_frames[i])
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel(f'Train\nLabel: {"Crash" if train_label == 0 else "Normal"}\nSeverity: {train_sev}', fontsize=12, rotation=0, ha='right', va='center')

    val_frames, val_label, val_sev = val_dataset[0]
    for i in range(min(3, len(val_frames))):
        axes[1, i].imshow(val_frames[i])
        axes[1, i].set_title(f'Frame {i+1}')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel(f'Validation\nLabel: {"Crash" if val_label == 0 else "Normal"}\nSeverity: {val_sev}', fontsize=12, rotation=0, ha='right', va='center')

    plt.tight_layout()
    plt.show()


visualize_sequence_samples()


# --- 12. TRAINING ---
for epoch in tqdm(range(EPOCHS)):
    # --- Train ---
    model.train()
    train_loss, correct = 0, 0
    train_bce_sum, train_ce_sum = 0, 0
    train_sev_correct, train_sev_total = 0, 0
    for pixel_values, labels, severities in tqdm(train_loader, total=len(train_loader)):
        pixel_values = pixel_values.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        severities = severities.to(DEVICE)

        optimizer.zero_grad()
        bin_logits, sev_logits = model(pixel_values)

        bce_loss = criterion(bin_logits, labels)
        crash_mask = (labels.squeeze(1) == 0) & (severities > 0)
        if crash_mask.any():
            sev_targets = (severities[crash_mask] - 1).long()
            ce_loss = severity_criterion(sev_logits[crash_mask], sev_targets)
            with torch.no_grad():
                sev_pred = sev_logits[crash_mask].argmax(dim=1)
                train_sev_correct += (sev_pred == sev_targets).sum().item()
                train_sev_total += sev_targets.numel()
        else:
            ce_loss = torch.zeros(1, device=DEVICE, dtype=bce_loss.dtype)

        loss = bce_loss + SEVERITY_LOSS_WEIGHT * ce_loss
        loss.backward()
        optimizer.step()

        batch_size = pixel_values.size(0)
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
        for pixel_values, labels, severities in val_loader:
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            severities = severities.to(DEVICE)
            bin_logits, sev_logits = model(pixel_values)

            bce_loss = criterion(bin_logits, labels)
            crash_mask = (labels.squeeze(1) == 0) & (severities > 0)
            if crash_mask.any():
                sev_targets = (severities[crash_mask] - 1).long()
                ce_loss = severity_criterion(sev_logits[crash_mask], sev_targets)
                sev_pred = sev_logits[crash_mask].argmax(dim=1)
                val_sev_correct += (sev_pred == sev_targets).sum().item()
                val_sev_total += sev_targets.numel()
            else:
                ce_loss = torch.zeros(1, device=DEVICE, dtype=bce_loss.dtype)

            loss = bce_loss + SEVERITY_LOSS_WEIGHT * ce_loss
            batch_size = pixel_values.size(0)
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
            'history': history.copy(),
            'config': {
                'num_frames': NUM_FRAMES,
                'checkpoint': TIMESFORMER_CHECKPOINT,
            }
        }, "best_checkpoint_timesformer.pth")

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
    'history': history,
    'config': {
        'num_frames': NUM_FRAMES,
        'checkpoint': TIMESFORMER_CHECKPOINT,
    }
}, "last_checkpoint_timesformer.pth")

# --- LOAD BEST CHECKPOINT ---
print(f"\ud83c\udfc6 Loading best checkpoint from epoch {best_epoch + 1} (Val Acc: {best_val_acc:.4f})")
best_checkpoint = torch.load("best_checkpoint_timesformer.pth", map_location=DEVICE)
model.load_state_dict(best_checkpoint['model_state_dict'])

# --- SAVE MODEL (final) ---
torch.save(model.state_dict(), "timesformer_crash_multitask.pth")
print("\ud83d\udcbe Model saved: timesformer_crash_multitask.pth")
print("\ud83d\udcbe Best checkpoint saved: best_checkpoint_timesformer.pth")
print("\ud83d\udcbe Last checkpoint saved: last_checkpoint_timesformer.pth")


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


