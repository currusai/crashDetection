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
    base, name = os.path.split(video_path)
    if not name.endswith("_motion_vectors.mp4"):
        raise ValueError(f"Unexpected video name pattern: {name}")
    prefix = name[: -len("_motion_vectors.mp4")]
    return os.path.join(base, f"{prefix}_annotations.csv")


def load_intervals(csv_path: str) -> List[Tuple[int, int]]:
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

    intervals.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            ms, me = merged[-1]
            merged[-1] = (ms, max(me, e))
    return merged


def detect_crash_box_top_right(frame: np.ndarray) -> bool:
    if frame is None or frame.size == 0:
        return False
    height, width = frame.shape[:2]

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

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)

    red_mask = (r >= 200) & (g <= 60) & (b <= 60)
    red_fraction = float(np.count_nonzero(red_mask)) / red_mask.size
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    red_dominance = mean_r - max(mean_g, mean_b)

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


def apply_interval_relaxation(
    pred_flags: List[int],
    intervals: List[Tuple[int, int]],
) -> List[int]:
    """
    For each annotated interval, if any frame within has pred==1, mark all frames
    in that interval as pred==1, else pred==0. Outside intervals, keep original preds.
    """
    adjusted = pred_flags.copy()
    n = len(pred_flags)
    for (s, e) in intervals:
        if s >= n:
            continue
        s_clamped = max(0, s)
        e_clamped = min(n - 1, e)
        if e_clamped < s_clamped:
            continue
        any_pos = any(pred_flags[i] == 1 for i in range(s_clamped, e_clamped + 1))
        fill_val = 1 if any_pos else 0
        for i in range(s_clamped, e_clamped + 1):
            adjusted[i] = fill_val
    return adjusted


def analyze_directory(results_dir: str, output_dir: str) -> None:
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
        print(f"Processing {video_name} with {len(intervals)} annotated crash interval(s) [interval-relaxed]...")

        pred_flags: List[int] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pred = 1 if detect_crash_box_top_right(frame) else 0
            pred_flags.append(pred)
        cap.release()

        num_frames = len(pred_flags)

        # Build true labels vector from intervals
        y_true_video = [0] * num_frames
        for (s, e) in intervals:
            if s >= num_frames:
                continue
            s_clamped = max(0, s)
            e_clamped = min(num_frames - 1, e)
            if e_clamped < s_clamped:
                continue
            for i in range(s_clamped, e_clamped + 1):
                y_true_video[i] = 1

        # Apply interval relaxation
        y_pred_video = apply_interval_relaxation(pred_flags, intervals)

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
    print("\nOverall metrics (interval-relaxed):")
    print(
        f"  frames={len(y_true_all)} | tp={int(overall['tp'])} tn={int(overall['tn'])} fp={int(overall['fp'])} fn={int(overall['fn'])} "
        f"acc={overall['accuracy']:.4f} prec={overall['precision']:.4f} rec={overall['recall']:.4f} f1={overall['f1']:.4f}"
    )

    cm_path = os.path.join(output_dir, "confusion_matrix_crash_box_interval_relaxed.png")
    save_confusion_matrix(
        tp=int(overall["tp"]),
        tn=int(overall["tn"]),
        fp=int(overall["fp"]),
        fn=int(overall["fn"]),
        out_path=cm_path,
    )
    print(f"Saved confusion matrix to: {cm_path}")

    summary_path = os.path.join(output_dir, "metrics_summary_interval_relaxed.txt")
    with open(summary_path, "w") as f:
        f.write("Interval-relaxed CRASH box detection metrics\n")
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
        description=(
            "Analyze CRASH box per frame with interval-relaxed evaluation: if any frame in an "
            "annotated interval is predicted crash, the whole interval's frames are treated as predicted crash."
        ),
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


