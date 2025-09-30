import argparse
import os
import sys
import json
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple, List

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
try:
    import torch
except Exception:
    torch = None


VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Export vehicle motion vectors (start/end points) per frame to JSONL.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--weights", type=str, default="yolo12x.pt", help="Path to YOLO weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device for inference (auto GPU/MPS), e.g. '0','cpu','mps'")
    parser.add_argument("--frame_limit", type=int, default=-1, help="Frames to process (-1 for full video)")
    parser.add_argument("--history", type=int, default=8, help="Previous frames to consider for motion vector")
    parser.add_argument("--out", type=str, default=None, help="Path to output JSONL (default next to input)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
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


def main():

    args = parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Input video not found: {args.video_path}")
        sys.exit(1)

    model = ensure_model(args.weights)
    tracker_yaml = "botsort.yaml"

    cap = cv2.VideoCapture(args.video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print("Failed to open video")
            sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Output path
    if args.out is None:
        root, _ = os.path.splitext(args.video_path)
        args.out = f"{root}_vectors.jsonl"

    # Track history of centers per id
    track_centers: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=max(2, args.history)))

    frame_idx = 0
    with open(args.out, "w") as f:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if args.frame_limit >= 0 and frame_idx >= args.frame_limit:
                    break

                # Auto-select device
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

                frame_vectors: List[dict] = []

                if results and len(results) > 0:
                    res = results[0]
                    if res.boxes is not None and res.boxes.shape[0] > 0:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        clss = res.boxes.cls.cpu().numpy().astype(int)
                        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

                        if ids is not None:
                            for i in range(len(boxes)):
                                class_id = int(clss[i])
                                if class_id not in VEHICLE_CLASS_IDS:
                                    continue
                                track_id = int(ids[i]) if i < len(ids) else None
                                if track_id is None or track_id < 0:
                                    continue
                                xyxy = boxes[i]
                                cx, cy = compute_center_xyxy(xyxy)
                                centers_deque = track_centers[track_id]
                                centers_deque.append((cx, cy))

                                if len(centers_deque) >= 2:
                                    # Weighted deltas across history
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

                                        start_point = (float(cx), float(cy))
                                        end_point = (float(cx + avg_dx * fps), float(cy + avg_dy * fps))

                                        frame_vectors.append({
                                            "track_id": int(track_id),
                                            "class_id": int(class_id),
                                            "frame_index": int(frame_idx),
                                            "start": {"x": start_point[0], "y": start_point[1]},
                                            "end": {"x": end_point[0], "y": end_point[1]},
                                        })

                # Write one JSON object per frame (list of vectors)
                f.write(json.dumps({
                    "video": args.video_path,
                    "frame_index": int(frame_idx),
                    "fps": float(fps),
                    "vectors": frame_vectors,
                }) + "\n")

                frame_idx += 1

        finally:
            cap.release()

    print(f"Saved motion vectors to: {args.out}")


if __name__ == "__main__":

    main()


