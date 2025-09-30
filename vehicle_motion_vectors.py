import argparse
import os
import sys
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple, List, Optional
import csv
import glob as globlib
import random
from datetime import datetime
import json
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, total=None, desc=None):
        return x

import cv2
import numpy as np
import subprocess
import shutil

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
try:
    import torch
except Exception:
    torch = None


# Vehicle class IDs for COCO (commonly used by YOLO):
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2, 3}


# Optional: set arguments directly in this script by filling this dict.
# If set to None, CLI args are used. If set to a dict, these values override defaults.
# Example below near the end of the file under the Usage section.
SCRIPT_ARGS = {
    "video_path": ".",                 # Positional; ignored if using --data_dir for interval/auto-tune
    "data_dir": ".",    # Where to find videos and *_annotations.csv
    "weights": "yolo12x.pt",
    "imgsz": 1280,
    "conf": 0.4,                       # Fixed during auto-tune
    "iou": 0.45,
    "device": None,                    # "0", "cpu", or "mps"
    "frame_limit": -1,                 # -1 = full length
    "history": 32,                     # Fixed during auto-tune
    "arrow_scale": 0.02,
    "arrow_color": "0,255,0",
    "thickness": 2,
    "save_path": None,                 # Outputs saved next to the script if None
    "show_ids": True,
    "verbose": True,

    # Crash scoring (only tuned if weight > 0)
    "crash_threshold": 0.9,            # Fixed during auto-tune
    "weight_predict": 1.0,
    "weight_intersect": 0.0,
    "weight_abrupt": 0.0,

    # Predict-related
    "pred_horizon": 8,
    "pred_step": 1,
    "pred_iou_thresh": 0.10,
    "intersect_margin": -12,           # Negative shrinks boxes
    "min_pair_speed": 3.0,
    "pred_viz_threshold": 0.2,

    # Abrupt-related
    "abrupt_angle_deg": 75.0,
    "min_speed": 1.0,
    "abrupt_memory": 1,

    # Interval/auto-tune
    "interval_mode": True,
    "glob": "Crashes*.mp4",
    "buffer_seconds": 1.0,
    "auto_tune": True,
    "tune_iters": 3,
    "experiment_root": "experiments",
}


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Detect, track, and draw motion vectors for vehicles.")
    parser.add_argument("video_path", type=str, help="Path to input video, or any path when using interval/auto-tune")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory to search for input videos and CSV annotation files")
    parser.add_argument("--weights", type=str, default="yolo12x.pt", help="Path to YOLO weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (auto-detects GPU/MPS), e.g. '0', 'cpu', or 'mps'")
    parser.add_argument("--frame_limit", type=int, default=1000, help="Number of frames to process (-1 for full video)")
    parser.add_argument("--history", type=int, default=8, help="How many previous frames to consider for motion vector")
    parser.add_argument("--arrow_scale", type=float, default=1.0, help="Scale factor for arrow length")
    parser.add_argument("--arrow_color", type=str, default="0,255,0", help="Arrow BGR color as 'B,G,R'")
    parser.add_argument("--thickness", type=int, default=2, help="Arrow thickness")
    parser.add_argument("--save_path", type=str, default=None, help="Output annotated video path (default next to input)")
    parser.add_argument("--show_ids", action="store_true", help="Draw tracker IDs on vehicles")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    # Crash detection configuration
    parser.add_argument("--crash_threshold", type=float, default=0.9, help="Weighted score threshold to flag crash")
    parser.add_argument("--weight_abrupt", type=float, default=0.0, help="Weight for abrupt direction change factor")
    parser.add_argument("--weight_predict", type=float, default=1, help="Weight for collision prediction factor")
    parser.add_argument("--weight_intersect", type=float, default=0.0, help="Weight for bbox intersection factor")
    parser.add_argument("--abrupt_angle_deg", type=float, default=75.0, help="Minimum direction change angle to consider abrupt (degrees)")
    parser.add_argument("--min_speed", type=float, default=1.0, help="Minimum per-frame speed (pixels) to consider direction change")
    parser.add_argument("--abrupt_memory", type=int, default=1, help="Frames to smooth velocity for abrupt detection (1 disables smoothing)")
    parser.add_argument("--pred_horizon", type=int, default=10, help="Frames ahead to predict for potential collision")
    parser.add_argument("--pred_step", type=int, default=1, help="Frame step when rolling prediction (>=1). Larger skips speed up")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.0, help="If >0, require IoU >= threshold for predicted collision")
    parser.add_argument("--intersect_margin", type=int, default=0.1, help="Margin (pixels) around boxes when checking intersection")
    parser.add_argument("--min_pair_speed", type=float, default=0.5, help="Min combined speed to consider prediction/intersection for a pair")
    parser.add_argument("--pred_viz_threshold", type=float, default=0.2, help="Min predicted collision factor to label a vehicle as 'PRED'")
    # Interval/Auto-tune features
    parser.add_argument("--interval_mode", action="store_true", help="Process only annotated crash intervals (with buffer) across matching videos and concatenate output")
    parser.add_argument("--glob", type=str, default="Crashes*.mp4", help="Glob for videos to include when using interval/auto-tune modes")
    parser.add_argument("--buffer_seconds", type=float, default=1.0, help="Seconds of buffer before/after each interval")
    parser.add_argument("--auto_tune", action="store_true", help="Run random search over parameters to maximize crash interval coverage")
    parser.add_argument("--tune_iters", type=int, default=20, help="Number of random configs to evaluate during auto-tune")
    parser.add_argument("--experiment_root", type=str, default="experiments", help="Directory to write tuning runs and best outputs")
    if SCRIPT_ARGS is not None:
        # Apply in-code overrides as parser defaults; CLI values still take precedence
        known_dests = {a.dest for a in parser._actions}
        defaults = {k: v for k, v in (SCRIPT_ARGS or {}).items() if k in known_dests}
        if defaults:
            parser.set_defaults(**defaults)
    return parser.parse_args()


def ensure_model(weights_path: str):

    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Please install it via requirements.txt")
    return YOLO(weights_path)


def compute_center_xyxy(xyxy: np.ndarray) -> Tuple[float, float]:

    x1, y1, x2, y2 = xyxy.tolist()
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def parse_bgr(color_str: str) -> Tuple[int, int, int]:

    try:
        b, g, r = [int(v) for v in color_str.split(",")]
        return b, g, r
    except Exception:
        return 0, 255, 0


def angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    # Returns angle in degrees between two 2D vectors
    x1, y1 = v1
    x2, y2 = v2
    n1 = np.hypot(x1, y1)
    n2 = np.hypot(x2, y2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = (x1 * x2 + y1 * y2) / (n1 * n2 + 1e-6)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cosang)))
    return ang


def boxes_intersect(a: np.ndarray, b: np.ndarray, margin: int = 0) -> bool:
    # a, b are xyxy
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1m = ax1 - margin
    ay1m = ay1 - margin
    ax2m = ax2 + margin
    ay2m = ay2 + margin
    bx1m = bx1 - margin
    by1m = by1 - margin
    bx2m = bx2 + margin
    by2m = by2 + margin
    inter_w = min(ax2m, bx2m) - max(ax1m, bx1m)
    inter_h = min(ay2m, by2m) - max(ay1m, by1m)
    return inter_w > 0 and inter_h > 0


def intersection_xyxy(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # Returns integer xyxy of intersection of two boxes, or None if no overlap
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = int(max(ax1, bx1))
    y1 = int(max(ay1, by1))
    x2 = int(min(ax2, bx2))
    y2 = int(min(ay2, by2))
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    return None


def draw_translucent_rect(frame: np.ndarray, rect: Tuple[int, int, int, int], color=(0, 0, 255), alpha: float = 0.35):
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def predict_collision_factor(
    a_box: np.ndarray,
    a_vel: Tuple[float, float],
    b_box: np.ndarray,
    b_vel: Tuple[float, float],
    horizon: int,
    margin: int,
    step: int = 1,
    iou_thresh: float = 0.0,
) -> float:
    # Linear prediction: shift boxes by t * velocity (pixels/frame). Return factor in [0,1]
    # with nearer-time collisions weighted higher.
    max_factor = 0.0
    for t in range(step, max(1, horizon) + 1, max(1, step)):
        ax = a_box.copy()
        bx = b_box.copy()
        ax[[0, 2]] = ax[[0, 2]] + a_vel[0] * t
        ax[[1, 3]] = ax[[1, 3]] + a_vel[1] * t
        bx[[0, 2]] = bx[[0, 2]] + b_vel[0] * t
        bx[[1, 3]] = bx[[1, 3]] + b_vel[1] * t
        intersects = boxes_intersect(ax, bx, margin=margin)
        if iou_thresh > 0.0:
            # Lazy import to avoid top clutter
            inter = min(ax[2], bx[2]) - max(ax[0], bx[0])
            inter_h = min(ax[3], bx[3]) - max(ax[1], bx[1])
            inter_area = max(0.0, inter) * max(0.0, inter_h)
            if inter_area > 0:
                area_a = max(0.0, (ax[2] - ax[0])) * max(0.0, (ax[3] - ax[1]))
                area_b = max(0.0, (bx[2] - bx[0])) * max(0.0, (bx[3] - bx[1]))
                union = area_a + area_b - inter_area + 1e-6
                iou = inter_area / union
            else:
                iou = 0.0
            intersects = intersects and (iou >= iou_thresh)

        if intersects:
            # Sooner collisions contribute more
            factor = 1.0 - (t - 1) / max(1, horizon)
            if factor > max_factor:
                max_factor = factor
    return max_factor


def read_annotations_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                s = int(r.get("frame_start_number") or r.get("start_frame") or r.get("start") or 0)
                e = int(r.get("frame_end_number") or r.get("end_frame") or r.get("end") or 0)
                sev = int(r.get("crash_severity") or r.get("severity") or 3)
                if e >= s:
                    rows.append((s, e, sev))
            except Exception:
                continue
    return rows


def find_videos_with_annotations(root: str, pattern: str) -> List[Tuple[str, str]]:
    root_dir = root
    if root_dir is None or len(str(root_dir).strip()) == 0:
        root_dir = os.getcwd()
    if os.path.isfile(root_dir):
        root_dir = os.path.dirname(os.path.abspath(root_dir))
    matches = []
    for vid_path in sorted(globlib.glob(os.path.join(root_dir, pattern))):
        if not os.path.isfile(vid_path):
            continue
        base, _ = os.path.splitext(vid_path)
        csv_path = f"{base}_annotations.csv"
        if os.path.isfile(csv_path):
            matches.append((vid_path, csv_path))
    return matches


def open_capture_preferring_ffmpeg(path: str) -> cv2.VideoCapture:
    cap_ff = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if cap_ff.isOpened():
        return cap_ff
    return cv2.VideoCapture(path)


def run_on_video_range(
    model,
    tracker_yaml: str,
    args: argparse.Namespace,
    video_path: str,
    start_frame: int,
    end_frame: int,
    concat_writer: Optional[cv2.VideoWriter],
    out_size: Optional[Tuple[int, int]],
) -> Tuple[bool, float]:
    cap = open_capture_preferring_ffmpeg(video_path)
    if not cap.isOpened():
        return False, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    track_centers: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=max(2, args.history)))
    track_last_velocity: Dict[int, Tuple[float, float]] = {}
    collision_in_interval = False
    frame_idx = max(0, start_frame)

    try:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            device_to_use = args.device
            if device_to_use is None:
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    device_to_use = "0"
                elif (
                    torch is not None
                    and hasattr(torch, "backends")
                    and hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    device_to_use = "mps"
                else:
                    device_to_use = "cpu"

            results = model.track(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device_to_use,
                persist=True,
                verbose=args.verbose,
                tracker=tracker_yaml,
            )

            if not results or len(results) == 0:
                if concat_writer is not None:
                    if out_size is not None and (frame.shape[1], frame.shape[0]) != out_size:
                        frame = cv2.resize(frame, out_size)
                    concat_writer.write(frame)
                frame_idx += 1
                continue

            res = results[0]
            if res.boxes is None or res.boxes.shape[0] == 0:
                if concat_writer is not None:
                    if out_size is not None and (frame.shape[1], frame.shape[0]) != out_size:
                        frame = cv2.resize(frame, out_size)
                    concat_writer.write(frame)
                frame_idx += 1
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids = None
            if res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)

            if ids is None:
                if concat_writer is not None:
                    if out_size is not None and (frame.shape[1], frame.shape[0]) != out_size:
                        frame = cv2.resize(frame, out_size)
                    concat_writer.write(frame)
                frame_idx += 1
                continue

            vehicles_info = []

            for i in range(len(boxes)):
                class_id = int(clss[i])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                track_id = int(ids[i]) if i < len(ids) else None
                if track_id is None or track_id < 0:
                    continue

                xyxy = boxes[i]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                cx, cy = compute_center_xyxy(xyxy)
                centers_deque = track_centers[track_id]
                centers_deque.append((cx, cy))

                if len(centers_deque) >= 2:
                    deltas: List[Tuple[float, float]] = []
                    weights: List[float] = []
                    centers_list = list(centers_deque)
                    for j in range(1, len(centers_list)):
                        prev = centers_list[j - 1]
                        cur = centers_list[j]
                        dx = cur[0] - prev[0]
                        dy = cur[1] - prev[1]
                        deltas.append((dx, dy))
                        weights.append(j)

                    deltas_np = np.array(deltas, dtype=np.float32)
                    weights_np = np.array(weights, dtype=np.float32)
                    if deltas_np.shape[0] > 0:
                        weighted = (deltas_np.T * weights_np).T
                        avg_dx, avg_dy = np.sum(weighted, axis=0) / (np.sum(weights_np) + 1e-6)

                        speed_dx = avg_dx * fps
                        speed_dy = avg_dy * fps

                        arrow_dx = int(speed_dx * args.arrow_scale)
                        arrow_dy = int(speed_dy * args.arrow_scale)

                        start_point = (int(cx), int(cy))
                        end_point = (int(cx + arrow_dx), int(cy + arrow_dy))

                        cv2.arrowedLine(
                            frame,
                            start_point,
                            end_point,
                            parse_bgr(args.arrow_color),
                            thickness=max(1, args.thickness),
                            tipLength=0.3,
                        )

                        vehicles_info.append({
                            "id": track_id,
                            "box": xyxy.copy(),
                            "vel": (float(avg_dx), float(avg_dy)),
                            "speed": float(np.hypot(avg_dx, avg_dy)),
                        })

                        prev_vel = track_last_velocity.get(track_id, (0.0, 0.0))
                        if args.abrupt_memory > 1 and len(centers_deque) > args.abrupt_memory:
                            k = min(args.abrupt_memory, len(centers_list) - 1)
                            sm_dx = 0.0
                            sm_dy = 0.0
                            for m in range(1, k + 1):
                                p1 = centers_list[-(m + 1)]
                                p2 = centers_list[-m]
                                sm_dx += (p2[0] - p1[0])
                                sm_dy += (p2[1] - p1[1])
                            sm_dx /= k
                            sm_dy /= k
                            avg_dx, avg_dy = sm_dx, sm_dy
                        ang = angle_between(prev_vel, (avg_dx, avg_dy))
                        is_fast_enough = np.hypot(avg_dx, avg_dy) >= args.min_speed and np.hypot(prev_vel[0], prev_vel[1]) >= args.min_speed
                        vehicles_info[-1]["abrupt"] = bool(is_fast_enough and ang >= args.abrupt_angle_deg)
                        track_last_velocity[track_id] = (float(avg_dx), float(avg_dy))

            abrupt_factor = 1.0 if any(v.get("abrupt", False) for v in vehicles_info) else 0.0

            predict_factor = 0.0
            intersect_factor = 0.0
            collision_highlight_rect: Optional[Tuple[int, int, int, int]] = None
            for a_idx in range(len(vehicles_info)):
                for b_idx in range(a_idx + 1, len(vehicles_info)):
                    va = vehicles_info[a_idx]
                    vb = vehicles_info[b_idx]
                    if (va["speed"] + vb["speed"]) < args.min_pair_speed:
                        continue
                    if boxes_intersect(va["box"], vb["box"], margin=args.intersect_margin):
                        intersect_factor = 1.0
                        inter_rect = intersection_xyxy(va["box"], vb["box"])  # raw intersection
                        if inter_rect is not None:
                            collision_highlight_rect = inter_rect
                    pc = predict_collision_factor(
                        va["box"], va["vel"], vb["box"], vb["vel"], args.pred_horizon, args.intersect_margin,
                        step=max(1, args.pred_step), iou_thresh=max(0.0, args.pred_iou_thresh)
                    )
                    if pc > predict_factor:
                        predict_factor = pc

            crash_score = (
                args.weight_abrupt * abrupt_factor
                + args.weight_predict * predict_factor
                + args.weight_intersect * intersect_factor
            )

            if crash_score >= args.crash_threshold:
                collision_in_interval = True
                # Draw red rectangle with white "CRASH" text at top-right
                text = "CRASH"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                pad = 10
                rect_w = tw + 2 * pad
                rect_h = th + baseline + 2 * pad
                x1 = width - rect_w - 10
                y1 = 10
                x2 = x1 + rect_w
                y2 = y1 + rect_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + pad, y1 + pad + th),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                # Draw translucent red highlight box at estimated collision area if available
                if collision_highlight_rect is not None:
                    draw_translucent_rect(frame, collision_highlight_rect, color=(0, 0, 255), alpha=0.35)

            if concat_writer is not None:
                if out_size is not None and (frame.shape[1], frame.shape[0]) != out_size:
                    frame = cv2.resize(frame, out_size)
                concat_writer.write(frame)

            frame_idx += 1
    finally:
        cap.release()

    return collision_in_interval, fps


def build_intervals_from_dataset(root: str, pattern: str, buffer_seconds: float) -> List[Dict[str, int]]:
    dataset = []
    vids = find_videos_with_annotations(root, pattern)
    for vid_path, csv_path in vids:
        try:
            cap = open_capture_preferring_ffmpeg(vid_path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        except Exception:
            fps = 30.0
        intervals = read_annotations_csv(csv_path)
        for (s, e, sev) in intervals:
            buf = int(round(buffer_seconds * fps))
            dataset.append({
                "video_path": vid_path,
                "start": max(0, s - buf),
                "end": e + buf,
                "severity": int(sev),
                "fps": float(fps),
            })
    return dataset


def print_dataset_stats(dataset_intervals: List[Dict[str, int]]):
    per_video_frames: Dict[str, int] = defaultdict(int)
    per_video_sev: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    total_frames = 0
    for itv in dataset_intervals:
        frames = max(0, int(itv["end"]) - int(itv["start"]) + 1)
        total_frames += frames
        vp = itv["video_path"]
        per_video_frames[vp] += frames
        sev = int(itv.get("severity", 3))
        per_video_sev[vp][sev] += 1

    print("Interval dataset summary (one iteration):")
    print(f"  Total frames to process: {total_frames}")
    for vp in sorted(per_video_frames.keys()):
        sev_map = per_video_sev[vp]
        sev_summary = ", ".join([f"s{k}:{sev_map[k]}" for k in sorted(sev_map.keys())])
        print(f"  {os.path.basename(vp)} -> frames: {per_video_frames[vp]}, severity: [{sev_summary}]")


def random_config_sample(base_args: argparse.Namespace) -> Dict[str, float]:
    cfg: Dict[str, float] = {}

    # Do NOT sample global fixed parameters during auto-tune
    # history, conf, crash_threshold remain fixed

    # Predict-related parameters (only if predict weight > 0)
    if getattr(base_args, "weight_predict", 0) > 0:
        cfg["pred_horizon"] = random.choice([4, 6, 8, 10, 12, 15, 20, 25, 30])
        cfg["pred_step"] = random.choice([1, 2])
        cfg["pred_iou_thresh"] = round(random.uniform(0.0, 0.25), 2)
        # Margin used inside prediction overlap as well
        cfg["intersect_margin"] = -random.choice([6, 8, 10, 12, 14, 16, 20, 25, 30])
        # Pair speed gate for predict/intersect
        cfg["min_pair_speed"] = round(random.uniform(1.0, 10.0), 2)

    # Intersect-only parameters (only if intersect weight > 0 and not already added)
    if getattr(base_args, "weight_intersect", 0) > 0 and getattr(base_args, "weight_predict", 0) <= 0:
        cfg.setdefault("intersect_margin", -random.choice([6, 8, 10, 12, 14, 16]))
        cfg.setdefault("min_pair_speed", round(random.uniform(0.5, 4.0), 2))

    # Abrupt-related parameters (only if abrupt weight > 0)
    if getattr(base_args, "weight_abrupt", 0) > 0:
        cfg["abrupt_angle_deg"] = random.choice([60.0, 70.0, 75.0, 80.0, 90.0, 100.0])
        cfg["min_speed"] = round(random.uniform(0.5, 3.0), 2)
        cfg["abrupt_memory"] = random.choice([1, 2, 3, 4])

    return cfg


def apply_config_to_args(args: argparse.Namespace, cfg: Dict[str, float]) -> argparse.Namespace:
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def score_for_intervals(model, tracker_yaml: str, args: argparse.Namespace, intervals: List[Dict[str, int]], save_video_path: Optional[str]) -> Tuple[float, Optional[str]]:
    writer: Optional[cv2.VideoWriter] = None
    out_size: Optional[Tuple[int, int]] = None
    out_fps: float = 30.0
    total_score = 0.0
    any_written = False

    if save_video_path is not None and len(intervals) > 0:
        first_vid = intervals[0]["video_path"]
        cap0 = open_capture_preferring_ffmpeg(first_vid)
        if cap0.isOpened():
            out_fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
            w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
            h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_size = (w0, h0)
        cap0.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video_path, fourcc, out_fps, out_size if out_size else (1280, 720))

    for itv in intervals:
        detected, _ = run_on_video_range(
            model,
            tracker_yaml,
            args,
            itv["video_path"],
            itv["start"],
            itv["end"],
            writer,
            out_size,
        )
        if detected:
            sev = int(itv.get("severity", 3))
            total_score += 1.0 if sev >= 3 else 0.5
            any_written = True

    if writer is not None:
        writer.release()

    return total_score, (save_video_path if any_written and save_video_path is not None else None)


def main():

    args = parse_args()
    arrow_color = parse_bgr(args.arrow_color)

    model = ensure_model(args.weights)

    tracker_yaml = "botsort.yaml"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.interval_mode or args.auto_tune:
        root_for_search = args.data_dir if args.data_dir is not None else args.video_path
        dataset_intervals = build_intervals_from_dataset(root_for_search, args.glob, args.buffer_seconds)[:3]
        if len(dataset_intervals) == 0:
            print("No videos with annotation CSVs found for the provided input.")
            sys.exit(1)

        if args.auto_tune:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(script_dir, args.experiment_root, f"exp_{ts}")
            os.makedirs(exp_dir, exist_ok=True)
            history_path = os.path.join(exp_dir, "tuning_history.txt")
            best_cfg = None
            best_score = -1.0

            # Print dataset stats once up front
            print_dataset_stats(dataset_intervals)

            # Fix global parameters for tuning as requested
            args.history = 32
            args.conf = 0.4
            args.crash_threshold = 0.9

            # Announce which parameters will be tuned based on weights
            tuned_params = []
            if args.weight_predict > 0:
                tuned_params += ["pred_horizon", "pred_step", "pred_iou_thresh", "intersect_margin", "min_pair_speed"]
            if args.weight_intersect > 0 and args.weight_predict <= 0:
                tuned_params += ["intersect_margin", "min_pair_speed"]
            if args.weight_abrupt > 0:
                tuned_params += ["abrupt_angle_deg", "min_speed", "abrupt_memory"]
            print("Auto-tune parameters:", ", ".join(tuned_params) if tuned_params else "(none; check weights)")

            with open(history_path, "w") as histf:
                for i in tqdm(range(int(args.tune_iters)), total=int(args.tune_iters), desc="Auto-tune"):
                    cfg = random_config_sample(args)
                    eval_args = argparse.Namespace(**vars(args))
                    eval_args = apply_config_to_args(eval_args, cfg)
                    score, _ = score_for_intervals(model, tracker_yaml, eval_args, dataset_intervals, save_video_path=None)
                    rec = {"iter": i + 1, "score": score, "config": cfg}
                    histf.write(json.dumps(rec) + "\n")
                    if args.verbose:
                        print(f"[iter {i+1}] score={score:.2f} cfg={json.dumps(cfg)}")
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg

            best_vid_path = os.path.join(exp_dir, "best_output.mp4")
            final_args = argparse.Namespace(**vars(args))
            final_args = apply_config_to_args(final_args, best_cfg or {})
            _, produced = score_for_intervals(model, tracker_yaml, final_args, dataset_intervals, save_video_path=best_vid_path)

            best_cfg_path = os.path.join(exp_dir, "best_config.txt")
            with open(best_cfg_path, "w") as f:
                json.dump({"best_score": best_score, "config": best_cfg, "output_video": produced}, f, indent=2)
            print(f"Auto-tune complete. Best score {best_score:.2f}. Files saved in: {exp_dir}")
            return

        else:
            out_dir = script_dir
            concat_path = args.save_path or os.path.join(out_dir, "intervals_concatenated_output.mp4")
            print_dataset_stats(dataset_intervals)
            score, outp = score_for_intervals(model, tracker_yaml, args, dataset_intervals, save_video_path=concat_path)
            print(f"Processed {len(dataset_intervals)} intervals across dataset. Output: {outp or 'n/a'}. Score (for reference): {score:.2f}")
            return

    def open_capture(path: str) -> cv2.VideoCapture:
        cap_ff = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if cap_ff.isOpened():
            return cap_ff
        return cv2.VideoCapture(path)

    def fourcc_to_str(fourcc_int: float) -> str:
        try:
            i = int(fourcc_int)
            return "".join([chr((i >> 8 * k) & 0xFF) for k in range(4)])
        except Exception:
            return ""

    def transcode_to_h264(input_path: str) -> str:
        # Save the transcoded file next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(script_dir, f"{base_name}_h264.mp4")
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            print("ffmpeg not found. Please install ffmpeg (e.g., brew install ffmpeg) or transcode manually.")
            return input_path
        cmd = [
            ffmpeg_path,
            "-y",
            "-hwaccel",
            "none",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Transcoded AV1 video to H.264: {out_path}")
            return out_path
        except subprocess.CalledProcessError as e:
            print("ffmpeg transcode failed. stderr:\n" + e.stderr.decode(errors="ignore"))
            return input_path

    # Try opening input; if AV1 or unreadable, transcode to H.264 and reopen
    video_path_to_use = args.video_path
    cap = open_capture(video_path_to_use)
    if not cap.isOpened():
        print("Failed to open video with default codecs. Attempting software transcode to H.264...")
        video_path_to_use = transcode_to_h264(args.video_path)
        cap = open_capture(video_path_to_use)
        if not cap.isOpened():
            print("Failed to open video after transcode.")
            sys.exit(1)
    else:
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fourcc_str = fourcc_to_str(fourcc).lower()
        # Heuristic: if codec is AV1 (often 'av01'), transcode for compatibility
        # if "av01" in fourcc_str or "av1" in fourcc_str:
        #     print("Detected AV1 codec. Transcoding to H.264 for compatibility...")
        #     cap.release()
        #     video_path_to_use = transcode_to_h264(args.video_path)
        #     cap = open_capture(video_path_to_use)
        #     if not cap.isOpened():
        #         print("Failed to open video after transcode.")
        #         sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer
    if args.save_path is None:
        # Save next to this script, not the input
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        ext = os.path.splitext(args.video_path)[1] or ".mp4"
        args.save_path = os.path.join(script_dir, f"{base_name}_motion_vectors{ext}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

    # History of centers per track_id
    track_centers: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=max(2, args.history)))
    # Last velocity per track (pixels/frame) to compare direction changes
    track_last_velocity: Dict[int, Tuple[float, float]] = {}

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.frame_limit >= 0 and frame_idx >= args.frame_limit:
                break

            # Run model with tracking
            # Auto-select device if not provided
            device_to_use = args.device
            if device_to_use is None:
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    device_to_use = "0"  # first CUDA GPU
                elif (
                    torch is not None
                    and hasattr(torch, "backends")
                    and hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    device_to_use = "mps"  # Apple Silicon
                else:
                    device_to_use = "cpu"

            results = model.track(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=device_to_use,
                persist=True,
                verbose=args.verbose,
                tracker=tracker_yaml,
            )

            if not results or len(results) == 0:
                writer.write(frame)
                frame_idx += 1
                continue

            res = results[0]
            if res.boxes is None or res.boxes.shape[0] == 0:
                writer.write(frame)
                frame_idx += 1
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            ids = None
            if res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)

            # Draw motion vectors for vehicles only
            if ids is None:
                # No tracking IDs yet, write frame as-is
                writer.write(frame)
                frame_idx += 1
                continue

            # Collect per-vehicle info for further analysis (collision prediction/intersection)
            vehicles_info = []

            for i in range(len(boxes)):
                class_id = int(clss[i])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                track_id = int(ids[i]) if i < len(ids) else None
                if track_id is None or track_id < 0:
                    continue

                xyxy = boxes[i]
                # Draw vehicle bbox and track id
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
                cx, cy = compute_center_xyxy(xyxy)
                centers_deque = track_centers[track_id]
                centers_deque.append((cx, cy))

                # Compute motion vector using multiple previous frames: from oldest to newest
                if len(centers_deque) >= 2:
                    # Weighted average of deltas across history
                    deltas: List[Tuple[float, float]] = []
                    weights: List[float] = []
                    centers_list = list(centers_deque)
                    for j in range(1, len(centers_list)):
                        prev = centers_list[j - 1]
                        cur = centers_list[j]
                        dx = cur[0] - prev[0]
                        dy = cur[1] - prev[1]
                        deltas.append((dx, dy))
                        weights.append(j)  # more recent segments get higher weight

                    deltas_np = np.array(deltas, dtype=np.float32)
                    weights_np = np.array(weights, dtype=np.float32)
                    if deltas_np.shape[0] > 0:
                        weighted = (deltas_np.T * weights_np).T
                        avg_dx, avg_dy = np.sum(weighted, axis=0) / (np.sum(weights_np) + 1e-6)

                        # Scale by FPS to correlate to speed. Pixel/frame * fps = pixel/sec
                        speed_dx = avg_dx * fps
                        speed_dy = avg_dy * fps

                        # Arrow length scaled
                        arrow_dx = int(speed_dx * args.arrow_scale)
                        arrow_dy = int(speed_dy * args.arrow_scale)

                        start_point = (int(cx), int(cy))
                        end_point = (int(cx + arrow_dx), int(cy + arrow_dy))

                        cv2.arrowedLine(
                            frame,
                            start_point,
                            end_point,
                            arrow_color,
                            thickness=max(1, args.thickness),
                            tipLength=0.3,
                        )

                        label_parts = []
                        if args.show_ids:
                            label_parts.append(f"ID {track_id}")

                        # Record per-frame velocity for downstream analysis (pixels/frame)
                        vehicles_info.append({
                            "id": track_id,
                            "box": xyxy.copy(),
                            "vel": (float(avg_dx), float(avg_dy)),
                            "speed": float(np.hypot(avg_dx, avg_dy)),
                        })

                        # Abrupt direction change factor (per-vehicle)
                        prev_vel = track_last_velocity.get(track_id, (0.0, 0.0))
                        # Optional simple smoothing for abrupt detection
                        if args.abrupt_memory > 1 and len(centers_deque) > args.abrupt_memory:
                            # average of last K deltas
                            k = min(args.abrupt_memory, len(centers_list) - 1)
                            sm_dx = 0.0
                            sm_dy = 0.0
                            for m in range(1, k + 1):
                                p1 = centers_list[-(m + 1)]
                                p2 = centers_list[-m]
                                sm_dx += (p2[0] - p1[0])
                                sm_dy += (p2[1] - p1[1])
                            sm_dx /= k
                            sm_dy /= k
                            avg_dx, avg_dy = sm_dx, sm_dy
                        ang = angle_between(prev_vel, (avg_dx, avg_dy))
                        is_fast_enough = np.hypot(avg_dx, avg_dy) >= args.min_speed and np.hypot(prev_vel[0], prev_vel[1]) >= args.min_speed
                        # Mark abrupt flag in the info for potential use/inspection
                        vehicles_info[-1]["abrupt"] = bool(is_fast_enough and ang >= args.abrupt_angle_deg)
                        if vehicles_info[-1]["abrupt"]:
                            label_parts.append("ABRUPT")
                        # Update last velocity for next frame
                        track_last_velocity[track_id] = (float(avg_dx), float(avg_dy))

                        # Render labels above the car
                        if label_parts:
                            label_text = " ".join(label_parts)
                            (lw, lh), lb = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            lx = int(xyxy[0])
                            ly = max(0, int(xyxy[1]) - 8)
                            cv2.rectangle(frame, (lx, ly - lh - 6), (lx + lw + 6, ly + 2), (0, 0, 0), -1)
                            cv2.putText(frame, label_text, (lx + 3, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Compute frame-level crash factors
            abrupt_factor = 1.0 if any(v.get("abrupt", False) for v in vehicles_info) else 0.0

            # Pairwise predicted collision and current intersection factors
            predict_factor = 0.0
            intersect_factor = 0.0
            collision_highlight_rect: Optional[Tuple[int, int, int, int]] = None
            for a_idx in range(len(vehicles_info)):
                for b_idx in range(a_idx + 1, len(vehicles_info)):
                    va = vehicles_info[a_idx]
                    vb = vehicles_info[b_idx]
                    # Skip very slow pairs to reduce noise
                    if (va["speed"] + vb["speed"]) < args.min_pair_speed:
                        continue
                    # Current intersection (with margin)
                    if boxes_intersect(va["box"], vb["box"], margin=args.intersect_margin):
                        intersect_factor = 1.0
                        # Red translucent highlight of intersection region (without margin)
                        inter_rect = intersection_xyxy(va["box"], vb["box"])  # raw intersection
                        if inter_rect is not None:
                            collision_highlight_rect = inter_rect
                        # Draw expanded intersection boxes for visualization
                        def draw_margin_box(box, color):
                            x1 = int(box[0]) - args.intersect_margin
                            y1 = int(box[1]) - args.intersect_margin
                            x2 = int(box[2]) + args.intersect_margin
                            y2 = int(box[3]) + args.intersect_margin
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        # draw_margin_box(va["box"], (0, 165, 255))  # orange
                        # draw_margin_box(vb["box"], (0, 165, 255))
                    # Predicted collision in the near future
                    pc = predict_collision_factor(
                        va["box"], va["vel"], vb["box"], vb["vel"], args.pred_horizon, args.intersect_margin,
                        step=max(1, args.pred_step), iou_thresh=max(0.0, args.pred_iou_thresh)
                    )
                    if pc > predict_factor:
                        predict_factor = pc
                    # Visualize projection lines (center to future center)
                    if pc >= args.pred_viz_threshold:
                        acx, acy = compute_center_xyxy(va["box"])  # current center
                        bcx, bcy = compute_center_xyxy(vb["box"])  # current center
                        # Project A and B by horizon frames
                        a_end = (int(acx + va["vel"][0] * args.pred_horizon), int(acy + va["vel"][1] * args.pred_horizon))
                        b_end = (int(bcx + vb["vel"][0] * args.pred_horizon), int(bcy + vb["vel"][1] * args.pred_horizon))
                        cv2.line(frame, (int(acx), int(acy)), a_end, (255, 0, 0), 2)  # blue line for A
                        cv2.line(frame, (int(bcx), int(bcy)), b_end, (0, 0, 255), 2)  # red line for B

                        # Estimate predicted collision point by intersecting projected lines (approx)
                        # Line param: P = P0 + t * v
                        def line_intersection(p1, v1, p2, v2):
                            x1, y1 = p1
                            x2, y2 = (p1[0] + v1[0], p1[1] + v1[1])
                            x3, y3 = p2
                            x4, y4 = (p2[0] + v2[0], p2[1] + v2[1])
                            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                            if abs(den) < 1e-6:
                                return None
                            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 * x2 - y1 * x2) * (x3 - x4))
                            return None
                        # Instead of full line intersection (which is noisy), mark midpoints of projected endpoints
                        pred_mid_x = int((a_end[0] + b_end[0]) / 2)
                        pred_mid_y = int((a_end[1] + b_end[1]) / 2)
                        cv2.circle(frame, (pred_mid_x, pred_mid_y), 6, (0, 0, 255), -1)
                        # Use a small square around predicted midpoint as highlight if not already intersecting now
                        if collision_highlight_rect is None:
                            s = 10
                            collision_highlight_rect = (pred_mid_x - s, pred_mid_y - s, pred_mid_x + s, pred_mid_y + s)

            crash_score = (
                args.weight_abrupt * abrupt_factor
                + args.weight_predict * predict_factor
                + args.weight_intersect * intersect_factor
            )

            # Draw score diagnostics (total and individual factors) at top-left
            diag_font = cv2.FONT_HERSHEY_SIMPLEX
            diag_scale = 0.6
            diag_th = 2
            t1 = f"Score={crash_score:.2f}"
            t2 = f"A={abrupt_factor:.2f} P={predict_factor:.2f} I={intersect_factor:.2f}"
            (t1w, t1h), t1b = cv2.getTextSize(t1, diag_font, diag_scale, diag_th)
            (t2w, t2h), t2b = cv2.getTextSize(t2, diag_font, diag_scale, diag_th)
            pad = 8
            box_w = max(t1w, t2w) + 2 * pad
            box_h = (t1h + t1b) + (t2h + t2b) + 3 * pad
            x1d, y1d = 10, 10
            x2d, y2d = x1d + box_w, y1d + box_h
            cv2.rectangle(frame, (x1d, y1d), (x2d, y2d), (0, 0, 0), thickness=-1)
            cv2.putText(frame, t1, (x1d + pad, y1d + pad + t1h), diag_font, diag_scale, (255, 255, 255), diag_th, cv2.LINE_AA)
            cv2.putText(frame, t2, (x1d + pad, y1d + 2 * pad + t1h + t2h), diag_font, diag_scale, (255, 255, 255), diag_th, cv2.LINE_AA)

            if crash_score >= args.crash_threshold:
                # Draw red rectangle with white "CRASH" text at top-right
                text = "CRASH"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                pad = 10
                rect_w = tw + 2 * pad
                rect_h = th + baseline + 2 * pad
                x1 = width - rect_w - 10
                y1 = 10
                x2 = x1 + rect_w
                y2 = y1 + rect_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + pad, y1 + pad + th),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
                # Draw translucent red highlight box at estimated collision area if available
                if collision_highlight_rect is not None:
                    draw_translucent_rect(frame, collision_highlight_rect, color=(0, 0, 255), alpha=0.35)

            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"Saved annotated video to: {args.save_path}")


if __name__ == "__main__":

    main()


"""
Usage examples
--------------

CLI (auto-detect device):
    python vehicle_motion_vectors.py "input.mp4" --weights yolo12x.pt --frame_limit 500 --history 8 --arrow_scale 0.02 --show_ids

CLI (force GPU 0):
    python vehicle_motion_vectors.py "input.mp4" --weights yolo12x.pt --device 0 --frame_limit -1

CLI (force CPU):
    python vehicle_motion_vectors.py "input.mp4" --weights yolo12x.pt --device cpu --frame_limit 100

Programmatic:
    from vehicle_motion_vectors import main
    # Equivalent to: python vehicle_motion_vectors.py input.mp4 --frame_limit 200
    # Use: subprocess or sys.argv to pass args programmatically
    # Example:
    import sys
    sys.argv = [
        "vehicle_motion_vectors.py",
        "input.mp4",
        "--weights", "yolo12x.pt",
        "--frame_limit", "200",
        "--arrow_scale", "0.02",
        "--show_ids",
    ]
    main()
"""

# Example of setting arguments directly in the script:
# --------------------------------------------------
# To use, set SCRIPT_ARGS above, e.g.:
# SCRIPT_ARGS = {
#     "video_path": "Crashes caught on Seattle traffic cameras 9!.mp4",
#     "weights": "yolo12x.pt",
#     "frame_limit": -1,  # -1 means full video
#     "history": 8,
#     "arrow_scale": 0.02,
#     "show_ids": True,
#     # Optional explicit device override: "0" for first CUDA GPU, "mps" for Apple, "cpu"
#     # "device": "0",
# }



# python vehicle_motion_vectors.py Crashes\ caught\ on\ Seattle\ traffic\ cameras\ 9\!.mp4 --weights yolo12x.pt \
#   --history 32 \
#   --pred_horizon 10 \
#   --pred_step 1 \
#   --intersect_margin -8 \
#   --min_pair_speed 1.5 \
#   --weight_predict 1 --weight_intersect 0 --weight_abrupt 0 \
#   --crash_threshold 0.9 \
#   --conf 0.4 \
#   --arrow_scale 0.02 --show_ids --frame_limit 3000 --verbose

# python vehicle_motion_vectors.py Crashes\ caught\ on\ Seattle\ traffic\ cameras\ 9\!.mp4 --weights yolo12x.pt \
#   --history 32 \
#   --pred_horizon 6 \
#   --pred_step 1 \
#   --intersect_margin -8 \
#   --min_pair_speed 1.5 \
#   --weight_predict 1 --weight_intersect 0 --weight_abrupt 0 \
#   --crash_threshold 0.9 \
#   --conf 0.4 \
#   --arrow_scale 0.02 --show_ids --frame_limit 3000 --verbose


# "pred_horizon": 15, "pred_step": 2, "pred_iou_thresh": 0.05, "intersect_margin": -10, "min_pair_speed": 0.56