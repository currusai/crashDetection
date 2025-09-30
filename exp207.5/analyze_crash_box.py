import os
import glob
import csv
from typing import List, Tuple, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt


def list_motion_vector_videos(results_dir: str) -> List[str]:
    pattern = os.path.join(results_dir, "*_motion_vectors.mp4")
    return sorted(glob.glob(pattern))


def corresponding_annotations_csv(video_path: str) -> str:
    # Map .../<prefix>_motion_vectors.mp4 -> .../<prefix>_annotations.csv
    base, name = os.path.split(video_path)
    if not name.endswith("_motion_vectors.mp4"):
        raise ValueError(f"Unexpected video name pattern: {name}")
    prefix = name[: -len("_motion_vectors.mp4")]
    return os.path.join(base, f"{prefix}_annotations.csv")


def load_intervals(csv_path: str) -> List[Tuple[int, int]]:
    """
    Read intervals [start, end] (inclusive) where crash_flag is true.
    The CSV columns are expected as: video_filename,frame_start_number,frame_end_number,crash_flag,...
    We ignore the video_filename field and any extra fields.
    """
    intervals: List[Tuple[int, int]] = []
    if not os.path.exists(csv_path):
        return intervals

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flag_raw = str(row.get("crash_flag", "")).strip()
            flag = flag_raw.lower() in {"true", "1", "yes", "y", "t", "on", "TRUE".lower()}
            if not flag:
                continue
            try:
                start = int(str(row.get("frame_start_number", "")).strip())
                end = int(str(row.get("frame_end_number", "")).strip())
            except Exception:
                continue
            if end < start:
                start, end = end, start
            intervals.append((start, end))

    # Merge overlapping/adjacent intervals to speed membership checks
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            ms, me = merged[-1]
            merged[-1] = (ms, max(me, e))
    return merged


def frame_in_crash_intervals(frame_idx: int, intervals: List[Tuple[int, int]]) -> bool:
    # Intervals are non-overlapping and sorted; do linear scan with pointer in caller if desired.
    # For simplicity, do binary search here.
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = intervals[mid]
        if frame_idx < s:
            hi = mid - 1
        elif frame_idx > e:
            lo = mid + 1
        else:
            return True
    return False


def detect_crash_box_top_right(frame: np.ndarray) -> bool:
    """
    Detect the filled red rectangle with white text "CRASH" at the top-right, using the
    exact drawing parameters provided. We measure the fraction of red pixels inside the
    expected rectangle region and decide based on thresholds.
    """
    if frame is None or frame.size == 0:
        return False

    height, width = frame.shape[:2]

    # Recreate rectangle geometry
    text = "CRASH"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 10
    rect_w = tw + 2 * pad
    rect_h = th + baseline + 2 * pad
    x1 = max(0, width - rect_w - 10)
    y1 = 10
    x2 = min(width, x1 + rect_w)
    y2 = min(height, y1 + rect_h)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Compute red dominance
    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    red_mask = (r >= 200) & (g <= 60) & (b <= 60)
    red_fraction = float(np.count_nonzero(red_mask)) / red_mask.size

    # Also evaluate mean red dominance to be robust to compression
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    red_dominance = mean_r - max(mean_g, mean_b)

    # Because white text cuts into the red area, avoid requiring too-high coverage.
    # Tune thresholds conservatively; adjust if needed.
    return (red_fraction >= 0.4 and red_dominance >= 30.0) or (red_fraction >= 0.55)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_confusion_matrix(tp: int, tn: int, fp: int, fn: int, out_path: str) -> None:
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    labels = ["Normal", "Crash"]

    plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_directory(
    results_dir: str,
    output_dir: str,
) -> None:
    videos = list_motion_vector_videos(results_dir)
    if not videos:
        print(f"No videos found matching *_motion_vectors.mp4 in: {results_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for vid_path in videos:
        ann_csv = corresponding_annotations_csv(vid_path)
        intervals = load_intervals(ann_csv)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Failed to open video: {vid_path}")
            continue

        video_name = os.path.basename(vid_path)
        print(f"Processing {video_name} with {len(intervals)} annotated crash interval(s)...")

        frame_idx = 0
        y_true_video: List[int] = []
        y_pred_video: List[int] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pred_crash = 1 if detect_crash_box_top_right(frame) else 0
            true_crash = 1 if frame_in_crash_intervals(frame_idx, intervals) else 0

            y_pred_video.append(pred_crash)
            y_true_video.append(true_crash)

            frame_idx += 1

        cap.release()

        # Aggregate
        y_true_all.extend(y_true_video)
        y_pred_all.extend(y_pred_video)

        # Per-video metrics
        mv = compute_metrics(y_true_video, y_pred_video)
        print(
            f"  frames={len(y_true_video)} | tp={int(mv['tp'])} tn={int(mv['tn'])} fp={int(mv['fp'])} fn={int(mv['fn'])} "
            f"acc={mv['accuracy']:.4f} prec={mv['precision']:.4f} rec={mv['recall']:.4f} f1={mv['f1']:.4f}"
        )

    # Overall metrics and confusion matrix
    overall = compute_metrics(y_true_all, y_pred_all)
    print("\nOverall metrics:")
    print(
        f"  frames={len(y_true_all)} | tp={int(overall['tp'])} tn={int(overall['tn'])} fp={int(overall['fp'])} fn={int(overall['fn'])} "
        f"acc={overall['accuracy']:.4f} prec={overall['precision']:.4f} rec={overall['recall']:.4f} f1={overall['f1']:.4f}"
    )

    cm_path = os.path.join(output_dir, "confusion_matrix_crash_box.png")
    save_confusion_matrix(
        tp=int(overall["tp"]),
        tn=int(overall["tn"]),
        fp=int(overall["fp"]),
        fn=int(overall["fn"]),
        out_path=cm_path,
    )
    print(f"Saved confusion matrix to: {cm_path}")

    # Save metrics summary
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Per-directory CRASH box detection metrics\n")
        f.write(f"Source: {results_dir}\n\n")
        f.write(
            "Overall: "
            + (
                f"frames={len(y_true_all)} tp={int(overall['tp'])} tn={int(overall['tn'])} fp={int(overall['fp'])} fn={int(overall['fn'])} "
                f"acc={overall['accuracy']:.6f} prec={overall['precision']:.6f} rec={overall['recall']:.6f} f1={overall['f1']:.6f}\n"
            )
        )
    print(f"Saved metrics summary to: {summary_path}")


def main():
    import argparse

    default_results_dir = (
        "/Users/yousefradwan/Library/CloudStorage/GoogleDrive-radwanf2025@gmail.com/"
        "My Drive/Yousef/OttonomiAI/CarAccidentVideos/exp207.5/results (1)"
    )

    parser = argparse.ArgumentParser(
        description="Analyze CRASH box presence per frame in motion vector videos and compute metrics.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results_dir,
        help="Directory containing *_motion_vectors.mp4 and *_annotations.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (PNG, summary). Defaults to results_dir.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    analyze_directory(results_dir=results_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()


