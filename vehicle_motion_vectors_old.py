import argparse
import os
import sys
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple, List

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
SCRIPT_ARGS = None  # e.g., {"video_path": "/path/video.mp4", "frame_limit": -1, ...}


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Detect, track, and draw motion vectors for vehicles.")
    parser.add_argument("video_path", type=str, help="Path to input video")
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
    if SCRIPT_ARGS is not None:
        # Merge provided overrides with parser defaults
        defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
        defaults.update(SCRIPT_ARGS)
        if not defaults.get("video_path"):
            raise ValueError("SCRIPT_ARGS must include 'video_path'")
        return argparse.Namespace(**defaults)
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


def intersection_xyxy(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, int, int] | None:
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


def main():

    args = parse_args()
    arrow_color = parse_bgr(args.arrow_color)

    if not os.path.isfile(args.video_path):
        print(f"Input video not found: {args.video_path}")
        sys.exit(1)

    model = ensure_model(args.weights)

    # Configure tracker to BoT-SORT explicitly (pass yaml path string)
    tracker_yaml = "botsort.yaml"

    def open_capture(path: str) -> cv2.VideoCapture:
        # Prefer FFMPEG backend if available to maximize codec support
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
        # Save next to the original input name, not the transcoded one
        root, ext = os.path.splitext(args.video_path)
        args.save_path = f"{root}_motion_vectors{ext}"
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
            collision_highlight_rect: Tuple[int, int, int, int] | None = None
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