import os
import glob
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch.nn.functional as F


# --- CONFIG ---

@dataclass
class CFG:
    data_dir: str = os.path.dirname(__file__)
    window_seconds: float = 3.0
    # If <= 0, defaults to ~fps/2
    stride_frames: int = 0
    target_frames: int = 32
    img_size: int = 128
    batch_size: int = 8
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_split: float = 0.2
    latent_dim: int = 256
    dropout: float = 0.1
    model_out: str = os.path.join(os.path.dirname(__file__), "frame_autoencoder.pth")
    embeddings_out: str = os.path.join(os.path.dirname(__file__), "frame_ae_embeddings.npz")
    log_file: str = os.path.join(os.path.dirname(__file__), "train_autoencoder_frames.log")
    
    # Triplet loss parameters
    triplet_margin: float = 5
    triplet_weight: float = 0.4  # Weight for triplet loss relative to reconstruction loss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("frames_autoencoder")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


LOGGER = init_logger(CFG.log_file)


def list_videos_with_annotations(data_dir: str) -> List[Tuple[str, str]]:
    vids = []
    for mp4 in glob.glob(os.path.join(data_dir, "*.mp4")):
        prefix = os.path.splitext(os.path.basename(mp4))[0]
        csv = os.path.join(data_dir, f"{prefix}_annotations.csv")
        if os.path.exists(csv):
            vids.append((mp4, csv))
    LOGGER.info(f"Discovered {len(vids)} videos with annotations in {data_dir}")
    return vids


def read_intervals(csv_path: str) -> List[Tuple[int, int]]:
    df = pd.read_csv(csv_path)
    # Try common column names
    start_col = None
    end_col = None
    for c in df.columns:
        lc = c.lower()
        if "frame_start" in lc:
            start_col = c
        if "frame_end" in lc:
            end_col = c
    if start_col is None or end_col is None:
        for c in df.columns:
            lc = c.lower()
            if start_col is None and (lc == "start" or lc == "start_frame"):
                start_col = c
            if end_col is None and (lc == "end" or lc == "end_frame"):
                end_col = c
    if start_col is None or end_col is None:
        LOGGER.error(f"CSV missing required columns: {csv_path}")
        raise ValueError(f"CSV {csv_path} missing frame start/end columns")
    rows: List[Tuple[int, int]] = []
    for _, r in df.iterrows():
        try:
            s = int(r[start_col])
            e = int(r[end_col])
            if e < s:
                s, e = e, s
            rows.append((s, e))
        except Exception:
            continue
    rows.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in rows:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    LOGGER.debug(f"Loaded {len(merged)} merged intervals from {os.path.basename(csv_path)}")
    return merged


def any_overlap(frames: List[int], intervals: List[Tuple[int, int]]) -> bool:
    if not intervals:
        return False
    for f in frames:
        for s, e in intervals:
            if s <= f <= e:
                return True
            if f < s:
                break
    return False


def video_meta(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), total


def read_resampled_frames(video_path: str, frame_indices: List[int], out_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    last_good: Optional[np.ndarray] = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if last_good is None:
                # create black frame
                arr = np.zeros((out_size, out_size, 3), dtype=np.uint8)
            else:
                arr = last_good.copy()
        else:
            arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            arr = cv2.resize(arr, (out_size, out_size), interpolation=cv2.INTER_AREA)
            last_good = arr
        frames.append(arr)
    cap.release()
    x = np.stack(frames, axis=0)  # (T, H, W, C)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))  # (T, C, H, W)
    return x


class FrameSequence3sDataset(Dataset):
    def __init__(self, data_dir: str, window_seconds: float, stride_frames: int, target_frames: int, img_size: int) -> None:
        self.samples: List[Tuple[str, List[int], int, float]] = []  # (video_path, frames, label, fps)
        self.target_frames = target_frames
        self.img_size = img_size

        pairs = list_videos_with_annotations(data_dir)
        total_windows = 0
        total_crash = 0
        for video_path, csv_path in pairs:
            try:
                fps, total = video_meta(video_path)
            except Exception:
                LOGGER.warning(f"Failed to read meta for {os.path.basename(video_path)}; defaulting to 30 FPS")
                fps, total = 30.0, 0
            if total <= 0:
                LOGGER.warning(f"Skipping {os.path.basename(video_path)} (no frames)")
                continue
            window_len = max(1, int(round(window_seconds * fps)))
            stride = stride_frames if stride_frames > 0 else max(1, int(round(fps / 2)))
            intervals = read_intervals(csv_path)
            max_start = max(0, total - window_len)
            per_video_windows = 0
            per_video_crash = 0
            for start in range(0, max_start + 1, stride):
                frames = list(range(start, start + window_len))
                label = 1 if any_overlap(frames, intervals) else 0  # 1=crash, 0=normal
                self.samples.append((video_path, frames, label, fps))
                per_video_windows += 1
                per_video_crash += int(label == 1)
            total_windows += per_video_windows
            total_crash += per_video_crash
            LOGGER.info(
                f"Built {per_video_windows} windows for {os.path.basename(video_path)} | FPS={fps:.2f} | window_len={window_len} | stride={stride} | crash={per_video_crash} ({(per_video_crash/max(1,per_video_windows))*100:.1f}%)"
            )

        if not self.samples:
            raise RuntimeError("No frame samples built. Ensure videos and *_annotations.csv exist in data_dir.")
        LOGGER.info(f"Total windows: {total_windows} | Crash windows: {total_crash} ({(total_crash/max(1,total_windows))*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, frames, label, fps = self.samples[idx]
        T = len(frames)
        idxs = np.linspace(0, T - 1, num=self.target_frames)
        idxs = np.round(idxs).astype(int)
        selected = [frames[i] for i in idxs]
        x = read_resampled_frames(video_path, selected, self.img_size)  # (T, C, H, W)
        y = np.array([label], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class FrameEncoder2D(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32, latent_dim: int = 256, img_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
        )
        # compute spatial size after 4 downsamples by 2
        ds = img_size // 16
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear((base_ch * 8) * ds * ds, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        z = self.proj(f)
        return z


class FrameDecoder2D(nn.Module):
    def __init__(self, out_ch: int = 3, base_ch: int = 32, latent_dim: int = 256, img_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        ds = img_size // 16
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, (base_ch * 8) * ds * ds),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.img_size = img_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        ds = self.img_size // 16
        x = self.fc(z)
        x = x.contiguous().reshape(b, 256, ds, ds)  # assuming base_ch=32 => base_ch*8=256
        x = self.deconv(x)
        return x


class PerFrameAutoencoder(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = FrameEncoder2D(in_ch=3, base_ch=32, latent_dim=latent_dim, img_size=img_size, dropout=dropout)
        self.decoder = FrameDecoder2D(out_ch=3, base_ch=32, latent_dim=latent_dim, img_size=img_size, dropout=dropout)

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D)
        return self.encoder(x)

    def decode_frames(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D) -> (B, C, H, W)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x_flat = x.contiguous().reshape(b * t, c, h, w)
        z_flat = self.encode_frames(x_flat)
        recon_flat = self.decode_frames(z_flat)
        recon = recon_flat.contiguous().reshape(b, t, c, h, w)
        z_seq = z_flat.contiguous().reshape(b, t, -1).mean(dim=1)  # (B, D)
        return z_seq, recon


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs).float()
    y = torch.cat(ys).long()
    return x, y


class TripletLoss(nn.Module):
    """Triplet loss for embedding separation."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) - batch of embeddings
            labels: (B,) - batch of labels (0=normal, 1=crash)
        Returns:
            triplet_loss: scalar tensor
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        # Compute pairwise squared distances without torch.cdist (better MPS support)
        # pairwise_dist(i,j) = ||ei - ej||_2
        # Use (ei^2 sum) + (ej^2 sum) - 2*ei·ej, clamp at 0, then sqrt
        x = embeddings
        x2 = (x * x).sum(dim=1, keepdim=True)  # (B,1)
        # (B,B) = x2 + x2^T - 2*X*X^T
        dist2 = x2 + x2.T - 2.0 * (x @ x.T)
        dist2 = torch.clamp(dist2, min=0.0)
        pairwise_dist = torch.sqrt(dist2 + 1e-8)
        
        # Create masks for positive and negative pairs
        labels_expanded = labels.unsqueeze(1)
        same_class = (labels_expanded == labels_expanded.T).float()
        diff_class = (labels_expanded != labels_expanded.T).float()
        
        # Remove diagonal (self-pairs)
        same_class.fill_diagonal_(0)
        diff_class.fill_diagonal_(0)
        
        # Find valid triplets: anchor-positive-negative
        # For each anchor, find closest positive and closest negative
        losses = []
        for i in range(embeddings.size(0)):
            # Get distances from anchor i to all others
            anchor_dists = pairwise_dist[i]
            
            # Find positive pairs (same class, excluding self)
            pos_mask = same_class[i] > 0
            if pos_mask.sum() == 0:
                continue  # No positive pairs for this anchor
                
            # Find negative pairs (different class)
            neg_mask = diff_class[i] > 0
            if neg_mask.sum() == 0:
                continue  # No negative pairs for this anchor
                
            # Get closest positive and closest negative
            pos_dists = anchor_dists[pos_mask]
            neg_dists = anchor_dists[neg_mask]
            
            if len(pos_dists) > 0 and len(neg_dists) > 0:
                closest_pos = pos_dists.min()
                closest_neg = neg_dists.min()
                
                # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
                loss = F.relu(closest_pos - closest_neg + self.margin)
                losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return torch.stack(losses).mean()


def build_splits(dataset: FrameSequence3sDataset, seed: int, val_split: float):
    # Per-video split
    video_ids = []
    for v, _frames, _y, _fps in dataset.samples:
        video_ids.append(os.path.basename(v))
    unique = list(sorted(set(video_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    val_n = max(1, int(len(unique) * val_split))
    val_set = set(unique[:val_n])
    train_idx = [i for i, (v, _f, _y, _fps) in enumerate(dataset.samples) if os.path.basename(v) not in val_set]
    val_idx = [i for i, (v, _f, _y, _fps) in enumerate(dataset.samples) if os.path.basename(v) in val_set]

    class _Subset(Dataset):
        def __init__(self, base: FrameSequence3sDataset, indices: List[int]) -> None:
            self.base = base
            self.indices = indices
        def __len__(self) -> int:
            return len(self.indices)
        def __getitem__(self, idx: int):
            return self.base[self.indices[idx]]

    return _Subset(dataset, train_idx), _Subset(dataset, val_idx)


def train_autoencoder_frames():
    set_seed(CFG.seed)
    LOGGER.info("Starting frames autoencoder training")
    LOGGER.info(
        f"Config: data_dir={CFG.data_dir} | window_seconds={CFG.window_seconds} | target_frames={CFG.target_frames} | img_size={CFG.img_size} | batch_size={CFG.batch_size} | epochs={CFG.epochs} | latent_dim={CFG.latent_dim}"
    )
    ds = FrameSequence3sDataset(
        data_dir=CFG.data_dir,
        window_seconds=CFG.window_seconds,
        stride_frames=CFG.stride_frames,
        target_frames=CFG.target_frames,
        img_size=CFG.img_size,
    )
    train_ds, val_ds = build_splits(ds, seed=CFG.seed, val_split=CFG.val_split)

    # Prefer MPS on Apple Silicon if available; else CUDA; else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    LOGGER.info(f"Device: {device}")

    # Use pin_memory only for CUDA
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False, collate_fn=collate_batch, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, drop_last=False, collate_fn=collate_batch, pin_memory=pin_mem)
    model = PerFrameAutoencoder(img_size=CFG.img_size, latent_dim=CFG.latent_dim, dropout=CFG.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, CFG.epochs))
    criterion = nn.MSELoss()
    triplet_criterion = TripletLoss(margin=CFG.triplet_margin)
    num_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model params: {num_params:,}")
    LOGGER.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    LOGGER.info(f"Triplet loss weight: {CFG.triplet_weight} | margin: {CFG.triplet_margin}")

    for epoch in range(1, CFG.epochs + 1):
        epoch_start = time.time()
        model.train()
        tr_loss = 0.0
        tr_recon_loss = 0.0
        tr_triplet_loss = 0.0
        tr_total = 0
        for xb, yb in tqdm(train_loader, desc="Training", total=len(train_loader)):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            z, recon = model(xb)
            
            # Reconstruction loss
            recon_loss = criterion(recon, xb)
            
            # Triplet loss for embedding separation
            triplet_loss = triplet_criterion(z, yb)
            
            # Combined loss
            total_loss = recon_loss + CFG.triplet_weight * triplet_loss
            
            total_loss.backward()
            optimizer.step()
            
            tr_loss += float(total_loss.item()) * xb.size(0)
            tr_recon_loss += float(recon_loss.item()) * xb.size(0)
            tr_triplet_loss += float(triplet_loss.item()) * xb.size(0)
            tr_total += int(xb.size(0))
        scheduler.step()

        model.eval()
        va_loss = 0.0
        va_recon_loss = 0.0
        va_triplet_loss = 0.0
        va_total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Validating", total=len(val_loader)):
                xb = xb.to(device)
                yb = yb.to(device)
                z, recon = model(xb)
                recon_loss = criterion(recon, xb)
                triplet_loss = triplet_criterion(z, yb)
                total_loss = recon_loss + CFG.triplet_weight * triplet_loss
                
                va_loss += float(total_loss.item()) * xb.size(0)
                va_recon_loss += float(recon_loss.item()) * xb.size(0)
                va_triplet_loss += float(triplet_loss.item()) * xb.size(0)
                va_total += int(xb.size(0))
        epoch_dur = time.time() - epoch_start
        LOGGER.info(
            f"Epoch {epoch:02d}/{CFG.epochs} | "
            f"recon={tr_recon_loss/max(1,tr_total):.4f} | triplet={tr_triplet_loss/max(1,tr_total):.4f} | "
            f"val_recon={va_recon_loss/max(1,va_total):.4f} | val_triplet={va_triplet_loss/max(1,va_total):.4f} | "
            f"time={epoch_dur:.1f}s"
        )

        # Free cached memory each epoch on MPS similar to CUDA
        try:
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    torch.save(model.state_dict(), CFG.model_out)
    LOGGER.info(f"Saved frame autoencoder to {CFG.model_out}")

    # Embeddings and linear probe
    def collect(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        zs, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                z, _ = model(xb)
                zs.append(z.cpu().numpy())
                ys.append(yb.cpu().numpy())
        return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)

    train_z, train_y = collect(train_loader)
    val_z, val_y = collect(val_loader)
    np.savez(CFG.embeddings_out, train_z=train_z, train_y=train_y, val_z=val_z, val_y=val_y)
    LOGGER.info(f"Saved embeddings to {CFG.embeddings_out} | train_z={train_z.shape} | val_z={val_z.shape}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_z, train_y.ravel())
    pred = clf.predict(val_z)
    acc = accuracy_score(val_y.ravel(), pred)
    LOGGER.info(f"Linear probe accuracy (val): {acc:.4f}")
    try:
        report = classification_report(val_y.ravel(), pred, target_names=["normal", "crash"]) 
        LOGGER.info("\n" + report)
    except Exception:
        pass


if __name__ == "__main__":
    train_autoencoder_frames()


