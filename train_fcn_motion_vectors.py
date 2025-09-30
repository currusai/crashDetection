import os
import json
import math
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


class CFG:
    """Editable configuration for training.

    Edit these values directly in-file when running from a notebook or script.
    """

    # Directory containing the motion vector jsonl files and their matching annotation CSVs
    data_dir: str = os.path.dirname(__file__)

    # Number of frames that form one training sample (sequence length)
    frames_per_sample: int = 3

    # Interval between frames inside a sample: 1 uses consecutive frames, 2 skips one between, etc.
    frame_interval: int = 1

    # Keep top-K vectors per frame ranked by vector magnitude; pad with zeros if fewer
    max_vectors_per_frame: int = 64

    # Batch size for data loaders
    batch_size: int = 64

    # Number of training epochs
    epochs: int = 10

    # Initial learning rate (AdamW)
    lr: float = 3e-4

    # Weight decay for AdamW optimizer
    weight_decay: float = 1e-4

    # Fraction of videos to reserve for validation (per-video split; frames do not mix across splits)
    val_split: float = 0.2

    # Random seed for reproducibility
    seed: int = 42

    # Output path for the saved confusion matrix image
    confusion_out: str = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_vectors_jsonl(jsonl_path: str) -> Dict[int, List[Dict]]:
    """Read motion vectors from a .jsonl file into a dict: frame_index -> list of vectors."""
    frame_to_vectors: Dict[int, List[Dict]] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frame_idx = int(rec.get("frame_index", 0))
            vectors = rec.get("vectors", []) or []
            frame_to_vectors[frame_idx] = vectors
    return frame_to_vectors


def read_crash_intervals(csv_path: str) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive frame intervals with crash_flag True."""
    df = pd.read_csv(csv_path)
    # Normalize possible column names
    start_col = None
    end_col = None
    flag_col = None
    for col in df.columns:
        lc = col.lower()
        if "frame_start" in lc:
            start_col = col
        elif "frame_end" in lc:
            end_col = col
        elif "crash_flag" in lc or "is_crash" in lc or "crash" == lc:
            flag_col = col
    if start_col is None or end_col is None or flag_col is None:
        raise ValueError(f"CSV {csv_path} does not contain required columns.")
    intervals: List[Tuple[int, int]] = []
    for _, row in df.iterrows():
        try:
            is_crash = bool(row[flag_col])
        except Exception:
            # Some CSVs may use strings for booleans
            is_crash = str(row[flag_col]).strip().lower() in {"true", "1", "yes", "y"}
        if is_crash:
            start = int(row[start_col])
            end = int(row[end_col])
            if end < start:
                start, end = end, start
            intervals.append((start, end))
    # Merge overlapping intervals to speed lookup
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def any_frame_in_intervals(frames: List[int], intervals: List[Tuple[int, int]]) -> bool:
    if not intervals:
        return False
    for frame in frames:
        # Binary search could be used; linear is okay given small count
        for s, e in intervals:
            if s <= frame <= e:
                return True
            if frame < s:
                break
    return False


def extract_vector_features(v: Dict) -> List[float]:
    sx = float(v.get("start", {}).get("x", 0.0))
    sy = float(v.get("start", {}).get("y", 0.0))
    ex = float(v.get("end", {}).get("x", 0.0))
    ey = float(v.get("end", {}).get("y", 0.0))
    dx = ex - sx
    dy = ey - sy
    mag = math.hypot(dx, dy)
    return [dx, dy, sx, sy, ex, ey, mag]


class MotionVectorsSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        frames_per_sample: int = 3,
        frame_interval: int = 1,
        max_vectors_per_frame: int = 64,
        min_frame_index: int = 0,
        max_frame_index: int = None,
    ) -> None:
        self.data_dir = data_dir
        self.frames_per_sample = frames_per_sample
        self.frame_interval = frame_interval
        self.max_vectors_per_frame = max_vectors_per_frame

        # Find all *_vectors.jsonl files
        pattern = os.path.join(data_dir, "*_vectors.jsonl")
        self.vector_files = sorted(glob.glob(pattern))

        if not self.vector_files:
            raise FileNotFoundError(f"No *_vectors.jsonl files found in {data_dir}")

        # Build samples: (file_idx, start_frame, frames_list, label)
        self.samples: List[Tuple[int, int, List[int], int]] = []
        self.file_frame_maps: List[Dict[int, List[Dict]]] = []
        self.file_crash_intervals: List[List[Tuple[int, int]]] = []
        self.file_names: List[str] = []

        for file_idx, vec_path in enumerate(self.vector_files):
            base = os.path.basename(vec_path)
            if not base.endswith("_vectors.jsonl"):
                continue
            prefix = base[: -len("_vectors.jsonl")]
            csv_path = os.path.join(data_dir, f"{prefix}_annotations.csv")
            if not os.path.exists(csv_path):
                # Skip files with no annotations
                continue

            frame_to_vectors = read_vectors_jsonl(vec_path)
            intervals = read_crash_intervals(csv_path)

            # Use a compact effective index that matches our appended lists
            effective_file_idx = len(self.file_frame_maps)
            self.file_frame_maps.append(frame_to_vectors)
            self.file_crash_intervals.append(intervals)
            self.file_names.append(prefix)

            if not frame_to_vectors:
                continue
            available_frames = sorted(frame_to_vectors.keys())
            first_frame = available_frames[0] if min_frame_index is None else max(available_frames[0], min_frame_index)
            last_frame = available_frames[-1] if max_frame_index is None else min(available_frames[-1], max_frame_index)

            # Generate sequences within range
            stride = 1  # advance by 1 frame start to maximize samples
            max_start = last_frame - (self.frames_per_sample - 1) * self.frame_interval
            for start in range(first_frame, max_start + 1, stride):
                frames = [start + i * self.frame_interval for i in range(self.frames_per_sample)]
                # Some frames may be missing in jsonl; treat missing as zero vectors
                label = 1 if any_frame_in_intervals(frames, intervals) else 0
                self.samples.append((effective_file_idx, start, frames, label))

        if not self.samples:
            raise RuntimeError("No training samples built. Check your data directory and file patterns.")

        # Pre-compute feature dimensionality
        self.features_per_vector = len(extract_vector_features({}))
        self.features_per_frame = self.max_vectors_per_frame * self.features_per_vector
        self.features_per_sample = self.frames_per_sample * self.features_per_frame

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_features(self, file_idx: int, frame_idx: int) -> np.ndarray:
        frame_map = self.file_frame_maps[file_idx]
        vectors = frame_map.get(frame_idx, [])
        if not vectors:
            return np.zeros((self.max_vectors_per_frame, self.features_per_vector), dtype=np.float32).reshape(-1)

        feats = []
        for v in vectors:
            feats.append(extract_vector_features(v))

        feats = np.asarray(feats, dtype=np.float32)
        # Rank by magnitude (last column)
        order = np.argsort(-feats[:, -1])
        feats = feats[order]

        if feats.shape[0] >= self.max_vectors_per_frame:
            feats = feats[: self.max_vectors_per_frame]
        else:
            pad = np.zeros((self.max_vectors_per_frame - feats.shape[0], feats.shape[1]), dtype=np.float32)
            feats = np.vstack([feats, pad])

        return feats.reshape(-1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, _start, frames, label = self.samples[idx]
        frame_feat_list = [self._frame_features(file_idx, f) for f in frames]
        x = np.concatenate(frame_feat_list, axis=0)
        y = np.array([label], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class SimpleFCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x = torch.stack(xs).float()
    y = torch.cat(ys).long()
    return x, y


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    labels = ["Normal", "Crash"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.tight_layout()
    fig.savefig(out_path)
    # Also display in notebook
    plt.show()


def main() -> None:
    set_seed(CFG.seed)

    dataset = MotionVectorsSequenceDataset(
        data_dir=CFG.data_dir,
        frames_per_sample=CFG.frames_per_sample,
        frame_interval=CFG.frame_interval,
        max_vectors_per_frame=CFG.max_vectors_per_frame,
    )

    # Per-video split: group by source video (prefix)/file
    unique_files = list({name for name in dataset.file_names})
    rng = random.Random(CFG.seed)
    rng.shuffle(unique_files)
    val_count = max(1, int(len(unique_files) * CFG.val_split))
    val_files = set(unique_files[:val_count])
    train_files = set(unique_files[val_count:])

    train_indices = [i for i, (file_idx, _s, _f, _y) in enumerate(dataset.samples) if dataset.file_names[file_idx] in train_files]
    val_indices = [i for i, (file_idx, _s, _f, _y) in enumerate(dataset.samples) if dataset.file_names[file_idx] in val_files]

    class _Subset(Dataset):
        def __init__(self, base: MotionVectorsSequenceDataset, indices: List[int]) -> None:
            self.base = base
            self.indices = indices
        def __len__(self) -> int:
            return len(self.indices)
        def __getitem__(self, idx: int):
            return self.base[self.indices[idx]]

    train_ds = _Subset(dataset, train_indices)
    val_ds = _Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    input_dim = dataset.features_per_sample
    model = SimpleFCN(input_dim=input_dim).to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, CFG.epochs))
    criterion = nn.CrossEntropyLoss()

    # Logging: explain data formation
    print("\nData formation:")
    print(f"- Frames per sample: {dataset.frames_per_sample}")
    print(f"- Frame interval: {CFG.frame_interval} (even sampling)")
    print(f"- Features per vector: {dataset.features_per_vector} [dx, dy, sx, sy, ex, ey, |v|]")
    print(f"- Top-K vectors per frame: {dataset.max_vectors_per_frame} (zero-padded if fewer)")
    print(f"- Input dim per sample: {dataset.features_per_sample}")
    print(f"- Train videos: {len(train_files)} | Val videos: {len(val_files)}")
    print(f"- Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Print a sample of training data before starting training
    sample_x, sample_y = dataset[0]
    print("Sample input shape:", sample_x.shape, "Sample label:", int(sample_y.item()))
    print("Features per vector:", dataset.features_per_vector, "Max vectors/frame:", dataset.max_vectors_per_frame)
    print("Frames/sample:", dataset.frames_per_sample, "Interval:", CFG.frame_interval)

    # Log model architecture
    print("\nModel architecture:\n", model)

    train_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    lrs: List[float] = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(xb.size(0))

        # Record LR before stepping the scheduler (current epoch LR)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        scheduler.step()
        train_acc = correct / max(1, total)
        train_loss = running_loss / max(1, total)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += int((preds == yb).sum().item())
                val_total += int(xb.size(0))
        val_acc = val_correct / max(1, val_total)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}/{CFG.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

    # Plot learning curve (loss and accuracy)
    epochs_range = list(range(1, CFG.epochs + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot learning rate schedule
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, lrs, label="LR")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR vs Epochs")
    plt.tight_layout()
    plt.show()

    # Evaluate and save confusion matrix
    y_true: List[int] = []
    y_pred: List[int] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(yb.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=["Normal", "Crash"]))
    plot_and_save_confusion_matrix(np.array(y_true), np.array(y_pred), CFG.confusion_out)
    print(f"Saved confusion matrix to {CFG.confusion_out}")


if __name__ == "__main__":
    main()


